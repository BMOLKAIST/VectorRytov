classdef Coordinates < matlab.mixin.Copyable
    %COORDINATES 이 클래스의 요약 설명 위치
    %   자세한 설명 위치
    
    properties
        Nx = [];
        Nz = [];
        x = [];
        y = [];
        z = [];

        ux = [];
        uy = [];
        uz = [];
        

        dx = [];
        dz = [];

        Lx = [];
        Lz = [];
        
        dux = [];
        duz = [];

    end

    methods
        
        function update_parameters(obj)

            obj.Lx = obj.Nx*obj.dx;
            obj.dux = 1/obj.Lx;
            obj.x = obj.dx*reshape((1:obj.Nx)-ceil((obj.Nx+1)/2),1,1,1,[]);
            obj.y = obj.dx*reshape((1:obj.Nx)-ceil((obj.Nx+1)/2),1,1,[]);
            obj.ux = obj.dux*reshape(ifftshift((1:obj.Nx)-ceil((obj.Nx+1)/2)),1,1,1,[]);
            obj.uy = obj.dux*reshape(ifftshift((1:obj.Nx)-ceil((obj.Nx+1)/2)),1,1,[]);

            if ~isempty(obj.Nz)
                obj.Lz = obj.Nz*obj.dz;
                obj.duz = 1/obj.Lz;
                obj.z = obj.dz*reshape((1:obj.Nz)-ceil((obj.Nz+1)/2),1,1,1,1,[]);
                obj.uz = obj.duz*reshape(ifftshift((1:obj.Nz)-ceil((obj.Nz+1)/2)),1,1,1,1,[]);
            end
        end

        function u = u_vec_2D(obj,uin,constant)
            u = cat(1,obj.ux+uin(1) + 0*obj.uy, 0*obj.ux + obj.uy+uin(2), obj.Qz(uin,constant)+uin(3));
        end

        function u = phi_vec_2D(obj,uin,constant)
            u = obj.u_vec_2D(uin,constant);
            u = cross(0*u+[0;0;1],u,1);
            norm_u = sqrt(sum(u.^2,1));
            u(:,1,squeeze(norm_u == 0)) = u(:,1,squeeze(norm_u == 0)) + [1;0;0];
            norm_u(norm_u == 0) = 1;
            u = u./norm_u;
        end

        function u = theta_vec_2D(obj,uin,constant)
            u = cross(obj.u_vec_2D(uin,constant),obj.phi_vec_2D(uin,constant),1);
            norm_u = sqrt(sum(u.^2,1));
            u = u./norm_u;
        end

        function u = linear_pol_2D(obj,uin,pol,constant)
            u = obj.u_vec_2D(uin,constant);
            u = cross(0*u+pol,u,1);
            u = cross(obj.u_vec_2D(uin,constant),u,1);
            norm_u = sqrt(sum(u.^2,1));
            u = u./norm_u;
            u = u.*pagectranspose(u);
        end

        function q = Uz(obj,constant)
            u0 = constant.mediumRI/constant.wavelength;
            q = real(sqrt(u0.^2-obj.ux.^2-obj.uy.^2));
            q(q==0)=1;
        end

        function q = Qz(obj,uin,constant)
            u0 = constant.mediumRI/constant.wavelength;
            q = real(sqrt(u0.^2-(obj.ux+uin(1)).^2-(obj.uy+uin(2)).^2))-uin(3);
            q(q==-uin(3))=1;
        end

        function mask = NAmask(obj,constant)
            mask = (obj.ux.^2+obj.uy.^2) < (constant.NA/constant.wavelength)^2;
        end

        function mask = SMALLmask(obj,constant)
            mask = (obj.ux.^2+obj.uy.^2) < (0.5*constant.NA/constant.wavelength)^2;
        end
    end
    
end

