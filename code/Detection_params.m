classdef Detection_params < matlab.mixin.Copyable
    properties
        corner1 = [];
        corner2 = [];
        num_angle = [];
        prefix1 = [];
        prefix2 = [];
        ismuxed = false;
        offset1 = [0 0];
        offset2 = [0 0];
    end
    
    methods
        function one_cam(obj,corner1,corner2,num_angle)
            obj.corner1 = corner1;
            obj.corner2 = corner2;
            obj.num_angle = num_angle;
            obj.ismuxed = true;
        end
        
        function two_cam(obj,prefix1,corner1,offset1,prefix2,corner2,offset2,num_angle)
            obj.prefix1 = prefix1;
            obj.prefix2 = prefix2;
            obj.corner1 = corner1;
            obj.corner2 = corner2;
            obj.num_angle = num_angle;
            obj.offset1 = offset1;
            obj.offset2 = offset2;
        end
    end
end

