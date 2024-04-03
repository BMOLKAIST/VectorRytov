function z = pagedet(x,y)
z = abs(pagemtimes(pagectranspose(x(1:3,1:2,:,:)),y(1:3,1:2,:,:)));
z = z(1,1,:,:)+z(2,2,:,:)-z(1,2,:,:)-z(2,1,:,:);
end

