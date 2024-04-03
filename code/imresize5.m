function W = imresize5(V,varargin)
    W = [];
    for i = 1:3
        for j = 1:3
            W(i,j,:,:,:) = imresize3(squeeze(V(i,j,:,:,:)),varargin{1});
        end
    end
end

