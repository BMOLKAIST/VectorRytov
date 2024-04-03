function W = imrotate5(V,varargin)
    W = [];
    for i = 1:3
        for j = 1:3
            W(i,j,:,:,:) = imrotate3(squeeze(V(i,j,:,:,:)),varargin{1},varargin{2},varargin{3},varargin{4});
        end
    end
end

