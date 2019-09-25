classdef DTWTrans < handle
    properties
        width
        ix
        iz
        orig
        final
    end
    methods
        function obj = DTWTrans(width, ix, iz, orig) 
            obj.width = width;
            obj.ix = ix;
            obj.iz = iz;
            obj.orig = orig;
            obj.final = zeros(length(orig),1);
        end
        function out = do(obj, din)
            seq = 1:length(din);
            seq = seq(obj.ix);
            din = din(obj.ix);
            out = zeros(1, length(obj.iz));
            for i = 1:length(obj.iz)
                out(i) = mean(din(cell2mat(obj.iz(i))));
%                tmp = din(cell2mat(obj.iz(i)));
%                tmp1 = cell2mat(obj.iz(i));
%                oo = obj.orig;
%                oo = oo(i);
%                if length(tmp1) ~= 1
%                    display(tmp1);
%                    oo = obj.orig;
%                    display(oo(i));
%                end
%                [min_val, match] = min(abs(tmp-oo));
%                if min_val > oo*0.1
%                    if length(tmp) == 1
%                        target_tmp = tmp;
%                    else
%                        target_tmp = tmp(match);
%                    end
%                    fprintf('Diff %f %0.2f%% | Index %d | Target %f | Seq %f\n', min_val, min_val/oo*100, i, oo, target_tmp);
%                end
                
%                out(i) = tmp(match);
%                obj.final(i) = seq(tmp1(match));
                
            end
        end
        function out = transform(obj, din)
            out = din(obj.final);
            diff = abs(out - obj.orig);
            for i = 1:length(out)
                if diff(i) > obj.orig(i) * 0.1
                    out(i) = obj.orig(i) * (1 + 0.05*rand());
                end
            end
        end
    end
end