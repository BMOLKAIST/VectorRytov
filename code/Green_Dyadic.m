function Green = Green_Dyadic(coordinates,constant, uin)
u = coordinates.u_vec_2D(uin, constant);
u_norm2 = sum(u.^2,1);

Green =  (eye(3)-u.*permute(u,[2,1,3,4])./u_norm2);

end

