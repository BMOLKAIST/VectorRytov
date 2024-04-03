classdef Polarization_params < matlab.mixin.Copyable
    properties
        pol1 = [];
        pol2 = [];

        pref1 = [];
        pref2 = [];

        pin_a = [];
        pin_b = [];

        
    end
    
    methods
        function one_cam(obj,pref1,pref2,pin_a,pin_b)
            obj.pref1 = pref1;
            obj.pref2 = pref2;
            obj.pin_a = pin_a;
            obj.pin_b = pin_b;
            obj.pol1 = eye(3);
            obj.pol2 = eye(3);
        end
        
        function two_cam(obj,pol1,pref1,pol2,pref2,pin_a,pin_b)
            obj.pref1 = pref1;
            obj.pref2 = pref2;
            obj.pin_a = pin_a;
            obj.pin_b = pin_b;
            obj.pol1 = pol1;
            obj.pol2 = pol2;
        end
    end
end
