function [Out, idx] = dtw_reverse(CurrTest, SysMean)
    [rows, width] = size(CurrTest);
    [~, cols] = size((SysMean));
    sample = CurrTest(1,:);
    [~, ix, iy] = dtw(sample, (SysMean));
    Out = zeros(cols, rows);
    iz = cell(1, cols);
    for j = 1:cols
        tmp = find(iy == j);
        iz{j} = tmp;        
    end
    a = DTWTrans(width, ix, iz, SysMean);
    for i = 1:rows
        curr = CurrTest(i,:);
        curr = a.do(curr);
        Out(:, i) = curr;
    end
    Out = Out';   
    idx = a;
end
