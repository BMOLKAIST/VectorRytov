function operand = convolve_Green(operand,coordinates,constant)

L=max(coordinates.Lx,coordinates.Lz);
N3 = ceil(2*sqrt(3)*L./[coordinates.dx coordinates.dx coordinates.dz])+2;

coordinates_padded = Coordinates();
coordinates_padded.Nx = N3(1);
coordinates_padded.Nz = N3(3);

coordinates_padded.dx = coordinates.dx;
coordinates_padded.dz = coordinates.dz;

coordinates_padded.update_parameters();

operand = padarray(operand,[0, 0, N3-[coordinates.Nx coordinates.Nx coordinates.Nz]],"post");
operand = fft(fft(fft(operand,N3(1),3),N3(1),4),N3(3),5);

ur = sqrt(coordinates_padded.ux.^2+coordinates_padded.uy.^2+coordinates_padded.uz.^2);
u0 = constant.mediumRI/constant.wavelength;

Green = (2*pi)^(-2)./(ur.^2-u0.^2).*(1-exp(1i*sqrt(3)*L*2*pi*u0).*(cos(sqrt(3)*L*2*pi*ur)-1i*u0.*sin(sqrt(3)*L*2*pi*ur)./(ur)));
Green(ur==0) = (2*pi)^(-2)/(-u0.^2).*(1-exp(1i*sqrt(3)*L*2*pi*u0).*(1+1i*2*pi*u0.*sqrt(3).*L));
Green(ur==u0) = 1i*(sqrt(3)*L/(2*2*pi*u0)-exp(1i*sqrt(3)*L*2*pi*u0)/(2*(2*pi*u0)^2)*sin(sqrt(3)*L*2*pi*u0));
u = cat(1,coordinates_padded.ux+zeros(1,1,N3(1),N3(1),N3(3)),coordinates_padded.uy+zeros(1,1,N3(1),N3(1),N3(3)),coordinates_padded.uz+zeros(1,1,N3(1),N3(1),N3(3)));
Green = Green.*(eye(3) - u.*permute(u,[2 1 3 4 5])/u0^2);
operand = pagemtimes(Green,operand);
operand = ifft(ifft(ifft(operand,N3(1),3),N3(1),4),N3(3),5);
operand = operand(:,:,1:coordinates.Nx,1:coordinates.Nx,1:coordinates.Nz);

end

