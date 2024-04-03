function V = rotateTensorField(V,ang)

for i = 1:3
    for j = 1:3
        V(i,j,:,:,:) = imrotate3(squeeze(V(i,j,:,:,:)),ang,[0 0 -1],'nearest','crop');
    end
end

ang = ang *pi/180;

R = [cos(ang) -sin(ang);sin(ang) cos(ang)];
R(3,3)=1;

V = pagemtimes(R,pagemtimes(V,R'));
end

