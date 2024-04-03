classdef FieldManager < matlab.mixin.Copyable

    
    properties
        field_in=[]; %3-by-1-by-Nx-by-Ny-by-N
        field_out=[];

        field_tensor = [];

        u_in = []; %3-by-1-by-N

        constant = [];
        coordinates = [];

        % parameters for field retrival
        ROI = [];
        target_pix_size = [];
        use_GPU = true;
        trim = 5;

        num_angle = 0;
    
    end
    
    methods
        function obj = FieldManager(constant)
            obj.constant = constant;
        end


        function getScatteringMatrix(obj,params)
            coordinatesM = copy(obj.coordinates);
            coordinatesM.dx = obj.constant.Magnification * obj.coordinates.dx;
            coordinatesM.update_parameters();
            constantM = copy(obj.constant);
            constantM.mediumRI = 1;
            obj.field_tensor = zeros(3,3,obj.coordinates.Nx,obj.coordinates.Nx,obj.num_angle);
            
            for ind = 1:obj.num_angle
                fprintf("Grouping fields... %d completed out of %d\n",ind,obj.num_angle)
                uin = mean(obj.u_in(:,[ind ind+2*obj.num_angle]),2);
                obj.u_in(:,ind) = uin;
                uinM = uin/obj.constant.Magnification;
                uinM(3) = sqrt(1/obj.constant.wavelength.^2 - uinM(1).^2 - uinM(2).^2);
                fielda = obj.field_out(:,1,:,:,ind)./obj.field_in(:,1,:,:,ind);
                fieldb = obj.field_out(:,1,:,:,ind+2*obj.num_angle)./obj.field_in(:,1,:,:,ind+2*obj.num_angle);

                phi = obj.coordinates.phi_vec_2D(uin,obj.constant);
                theta = obj.coordinates.theta_vec_2D(uin,obj.constant);
                phiM = coordinatesM.phi_vec_2D(uinM,constantM);
                thetaM = coordinatesM.theta_vec_2D(uinM,constantM);

                M = pagemtimes(thetaM,pagetranspose(theta)) + pagemtimes(phiM,pagetranspose(phi));

                if size(params.pol1,2) == 1
                    P1 = coordinatesM.linear_pol_2D(uinM,params.pol1,constantM);
                else
                    P1 = eye(3);
                end
                if size(params.pol2,2) == 2
                    P2 = coordinatesM.linear_pol_2D(uinM,params.pol2,constantM);
                else
                    P2 = eye(3);
                end

                pin_a = phi(:,:,1,1).*params.pin_a(1) + theta(:,:,1,1).*params.pin_a(2);
                pin_b = phi(:,:,1,1).*params.pin_b(1) + theta(:,:,1,1).*params.pin_b(2);

                gphase1a = params.pref1'*P1(:,:,1,1)*M(:,:,1,1)*pin_a;
                gphase1b = params.pref1'*P1(:,:,1,1)*M(:,:,1,1)*pin_b;
                gphase2a = params.pref2'*P2(:,:,1,1)*M(:,:,1,1)*pin_a;
                gphase2b = params.pref2'*P2(:,:,1,1)*M(:,:,1,1)*pin_b;

                if ind == 1
                    figure;imagesc(angle(squeeze(fielda(1,1,:,:)))); axis image; colormap gray; colorbar
                    title('Select a background area')
                    r = drawrectangle;
                    bg = r.Position;
                    bg = [round(bg(2)), round(bg(2))+round(bg(4))-1, round(bg(1)), round(bg(1))+round(bg(3))-1 ];
                end
                fielda = fielda./mean(fielda(:,1,bg(1):bg(2),bg(3):bg(4)),[3 4]).*[gphase1a;gphase2a];
                fieldb = fieldb./mean(fieldb(:,1,bg(1):bg(2),bg(3):bg(4)),[3 4]).*[gphase1b;gphase2b];

                pagefft2 = @(x) fft(fft(x,[],3),[],4);
                pageifft2 = @(x) ifft(ifft(x,[],3),[],4);

                fielda = pagefft2(fielda); fielda(:,2,:,:) = pagefft2(fieldb);
                
                G = pagemtimes(pagemtimes(params.pref1',P1),M);
                G(2,:,:,:) = pagemtimes(pagemtimes(params.pref2',P2),M);
                [A,B,C] = pagesvd(G);

                B=B.*circshift(obj.coordinates.NAmask(obj.constant),[0 0 -round(uin(2)/obj.coordinates.dux) -round(uin(1)/obj.coordinates.dux)]);
                B(B~=0) = 1./B(B~=0);
                G = pagemtimes(pagemtimes(C,pagectranspose(B)),pagectranspose(A));
                H = inv([pin_a pin_b uin]);

                fielda = pageifft2(pagemtimes(G,fielda));
                fielda(:,3,:,:) = zeros(size(fielda(:,1,:,:))) + uin;
                obj.field_tensor(:,:,:,:,ind) = pagemtimes(fielda,H);
                % subplot(1,2,1);imagesc(squeeze(angle(obj.field_tensor(1,1,:,:,ind)+1i*obj.field_tensor(1,2,:,:,ind))-angle(obj.field_tensor(2,1,:,:,ind)+1i*obj.field_tensor(2,2,:,:,ind))));colorbar
                % subplot(1,2,2);imagesc(squeeze(angle(obj.field_tensor(1,1,:,:,ind)-1i*obj.field_tensor(1,2,:,:,ind))-angle(obj.field_tensor(2,1,:,:,ind)-1i*obj.field_tensor(2,2,:,:,ind))));colorbar

                % Illumination polarizations become orthogonal to the detection
                % polarizations
                %%imagesc(squeeze(abs(fielda(1,1,:,:)./mean(fielda(1,1,1:40,1:40),'all')-fieldb(1,1,:,:)./mean(fieldb(1,1,1:40,1:40),'all'))))
                %%imagesc(squeeze(abs(fielda(2,1,:,:)./mean(fielda(2,1,1:40,1:40),'all')-fieldb(2,1,:,:)./mean(fieldb(2,1,1:40,1:40),'all'))))

                % Detection polarizations become orthogonal to the
                % illumination polarizations
                %%imagesc(squeeze(angle(fielda(1,1,:,:)./mean(fielda(1,1,1:40,1:40),'all')-fielda(2,1,:,:)./mean(fielda(2,1,1:40,1:40),'all'))))
                %%imagesc(squeeze(angle(fieldb(1,1,:,:)./mean(fieldb(1,1,1:40,1:40),'all')-fieldb(2,1,:,:)./mean(fieldb(2,1,1:40,1:40),'all'))))
            end
        end
    end
end
