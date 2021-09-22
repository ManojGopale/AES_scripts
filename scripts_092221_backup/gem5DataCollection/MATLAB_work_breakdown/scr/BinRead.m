% This version reads the binary input of unsigned char sizes.


function [ix, iy, data_out] = BinRead(~, num, folder_name, comp, ix_in, iy_in, display)

% Arguments:
% ~: ignored
% num: target key index. Starts from 0 
% folder_name: a cell array of strings each containing the data folders' (e.g. ["2019XXX","2019xxx","2019xxx"]
% comp: the system component string you would like to read, should be one of 'cpu'/'icache'/'dcache'/'l2cache'/'dram'
% ix_in, iy_in: the indexing arrays when aligning the traces. If not given a new set of arrays will be generated as output and can be use for other components alignment
% display: valid when all arguments are supplied. Enabling messages logging

%Returned values
% ix,iy: the indexing arrays. If ix_in and iy_in are given then the same arrays are returned
% data_out: data after auto alignment


% Default displays reading and writing info
if(nargin == 4 || nargin == 6)
    display = true;
end

% If specified display value is set
if(nargin == 5)
    display = ix_in;
end

if(display == true)
    fprintf('        Preparing inputs for %s %d... \n', comp, num);
end
%Mprefix = '/xdisk/bozhiliu/q_result/';
prefix = '/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/q_result/';
% Build the matrix that contains info on the length/folder pair of first
% occurence of a particular length trace
data = [];
trace = {};
for ff = 1:length(folder_name)
    folder = folder_name{ff};
    tmp = strcat(prefix, folder);
    tmp1 = load(sprintf("%s/Count%d.txt",tmp,num));
    curr = unique(tmp1);
    for i = 1:length(curr)
        if(length(data) == 0 || isempty(find(data(:,1) == curr(i))))
            data = [data ; curr(i), ff];
            fid = fopen(sprintf("%s/%s%d.bin", tmp,comp,num), "r");
            occur = find(tmp1 == curr(i), 1, 'first');
            % CPU data comes in two unsigned char per time slot
            if(contains(comp, 'cpu', 'IgnoreCase', true))
                read_coeff = 2;
                bin_format = 'unsigned char';
            elseif (contains(comp, 'dram', 'IgnoreCase', true))
                read_coeff = 1;
                bin_format = 'double';
            else
                read_coeff = 1;
                bin_format = 'unsigned char';
            end
            
            % adhoc fix: read dram data in ascii format
            if(contains(comp, 'dram','IgnoreCase',true)) && false
                din = csvread(sprintf("%s/%s%d.txt_out.csv", tmp,comp,num));
                din = din(1:sum(tmp1(1:occur)),1);
            else
                din = fread(fid, [sum(tmp1(1:occur))*read_coeff,1], bin_format);
                fclose(fid);
            end
            
            % Convert from to actual numbers except for dram
            din = din(length(din)-curr(i)*read_coeff+1:length(din));
            
            trace{length(trace)+1,1} = process(din, curr(i), comp);
            
        end
    end
end
[data, idx] = sortrows(data);
trace = trace(idx);


% Comment out: use dtw_reverse instead of dtw now
% Use DTW to build index transforms that from the lowest to the
% highest lengths traces
%if nargin == 4 || nargin == 5
%    ix = cell(length(data(:,1))-1,1);
%    iy = cell(length(data(:,1))-1,1);
%    curr_trace = cell2mat(trace(1));
%    for i = 1:length(data(:,1))-1
%       Everyone goes traverses the tree to the root node        
%        [~, ix_curr, iy_curr] = dtw(curr_trace, cell2mat(trace(i+1)));
%        ix{i} = ix_curr;
%        iy{i} = iy_curr;
%        curr_trace = curr_trace(ix_curr);
%    end
%elseif (nargin == 6 || nargin == 7)
%    ix = ix_in;
%    iy = iy_in;
%end

%trans = cell(length(data(:,1)),1);
trans = DTWTrans.empty();
if nargin == 4 || nargin == 5    
    for i = 1:length(data(:,1))
        curr_trace = cell2mat(trace(i));
        [dtmp, idx_tmp] = dtw_reverse(curr_trace, cell2mat(trace(1)));        
        trans(i) = idx_tmp;
    end
    ix = trans;
    iy = trans;
elseif nargin == 6 || nargin == 7
    ix = ix_in;
    iy = iy_in;
    trans = ix_in;
end


% Start to read all the data in and enlarge them on the fly
% Comment out: this will do a tree structure mapping that traverses up
% the tree to the top most node

%trans = zeros(length(data(:,1)), length(cell2mat(ix(end))));
%for curr = 1:length(data(:,1))
%    curr_data = 1:data(curr,1);
%    for j = curr:length(data(:,1))
%        if (j == curr)
%            if (j == 1)
%                curr_data = curr_data(cell2mat(ix(j)));
%            else
%                curr_data = curr_data(cell2mat(iy(j-1)));
%            end
%        else
%            curr_data = curr_data(cell2mat(ix(j-1)));
%        end
%    end
%    trans(curr,:) = curr_data;
%end


data_out = zeros(10000*length(folder_name), data(1,1))';
pos = 1;

%textprogressbar(sprintf('            Reading inputs for %s %d... ', comp, num));
for ff = 1:length(folder_name)
    folder = folder_name{ff};
    tmp = strcat(prefix, folder);
    count = load(sprintf("%s/Count%d.txt", tmp, num));
    fid = fopen(sprintf("%s/%s%d.bin", tmp, comp, num), "r");
    if(contains(comp, 'dram', 'IgnoreCase', true)) && false
        din = csvread(sprintf("%s/%s%d.txt_out.csv", tmp,comp,num));
    else
        din = fread(fid, bin_format);
    end
    
    curr_pos = 1;
%    diff_average = zeros(1,length(count));
    for i = 1:length(count)
        %        textprogressbar(pos/(length(folder_name)*100.0));
        %Mfprintf("%s: %d, %s\n", folder, i, comp);
        %Mfprintf("%d, %d\n", curr_pos, curr_pos+count(i)*read_coeff-1);
        curr_data = process(din(curr_pos:curr_pos+count(i)*read_coeff-1), count(i), comp);
        curr_pos = curr_pos + count(i)*read_coeff;
        order = find(data(:,1) == length(curr_data),1);
        %curr_data = curr_data(cell2mat(trans(order,:)));
        %curr_data = curr_data((trans(order,:)));
        if order ~= 1
            curr_func = trans(order);   
            curr_data = curr_func.do(curr_data);        
            %diff_average(i) = curr_diff;
        end
        data_out(:,pos) = curr_data;
        pos = pos+1;
    end
%    diff_average = mean(diff_average);
    if(~contains(comp, 'dram', 'IgnoreCase', true))
        fclose(fid);
    end
end
data_out = data_out';
if(display ==  true)
    fprintf('        Done reading inputs for %s %d.... \n', comp, num);
end
%textprogressbar('Done');
end


function val = interpolate(low, high, flip)
diff = high - low;
diff = (diff / 32) * flip;
val = low + diff;
end


function din_full = process(din, curr, comp)
din_full = zeros(1, curr);
if(contains(comp, 'dram', 'IgnoreCase', true))
    din_full = din;
    return;
end

if(contains(comp, 'cpu', 'IgnoreCase', true))
    for i = 1:2:length(din)
        switch(din(i))
            case 0 % Op_and
                din_full((i+1)/2) = interpolate(0.0951995,0.096490625,din(i+1));
            case 1 % bhi
                din_full((i+1)/2) = interpolate(0.073566438,0.074564125, din(i+1));
            case 2 % cmps
                din_full((i+1)/2) = interpolate(0.082291375,0.083407625, din(i+1));
            case 3 % orr
                din_full((i+1)/2) = interpolate(0.0951995,0.096490625, din(i+1));
            case 4 % subs
                din_full((i+1)/2) = interpolate(0.088788063,0.089992438,din(i+1));
            case 5 %eor
                din_full((i+1)/2) = interpolate(0.096497675,0.097806406,din(i+1));
            case 6 %strbcs
                din_full((i+1)/2) = interpolate(0.099615857,0.100967125,din(i+1));
            case 7 %smull
                din_full((i+1)/2) = interpolate(0.168913875,0.171205125,din(i+1));
            case 8 %mov
                din_full((i+1)/2) = interpolate(0.0951995,0.096490625,din(i+1));
            case 9 % rsb
                din_full((i+1)/2) = interpolate(0.0951995,0.096490625,din(i+1));
            case 10 % ldrb
                din_full((i+1)/2) = interpolate(0.071932,0.072864,din(i+1));
            case 11 % ldr
                din_full((i+1)/2) = interpolate(0.103311688,0.10465,din(i+1));
            case 12 % add
                din_full((i+1)/2) = interpolate(0.09087225,0.092105688,din(i+1));
            case 13 % subne
                din_full((i+1)/2) = interpolate(0.088708625,0.089992438,din(i+1));
            case 14 % subscs
                din_full((i+1)/2) = interpolate(0.088708625,0.089911719,din(i+1));
            case 15 % str
                din_full((i+1)/2) = interpolate(0.070369,0.071306,din(i+1));
            case 16 % strb
                din_full((i+1)/2) = interpolate(0.070434,0.071371,din(i+1));
            case 17 % tsts
                din_full((i+1)/2) = interpolate(0.084456938,0.085602563,din(i+1));
            case 18 % sub
                din_full((i+1)/2) = interpolate(0.09087225,0.092104688,din(i+1));
            case 19 % b
                din_full((i+1)/2) = interpolate(0.069768,0.069768,din(i+1));
            case 128 % empty
                din_full((i+1)/2) = 0;
        end
    end
elseif(contains(comp, 'dram', 'IgnoreCase', true))
    din_full = din_tmp;
elseif(contains(comp, 'icache', 'IgnoreCase', true))
    for i = 1:length(din)
        switch(din(i))
            case 0 % icache hit
                din_full(i) = 0.0000001*1000;
            case 1 % icache_miss
                din_full(i) = 0.00000042*1000;
            case 2 % icache_idle
                din_full(i) = 0;
            case 3 % icache_miss_end
                din_full(i) = 0.00000042*1000;
        end
    end
elseif(contains(comp, 'dcache', 'IgnoreCase', true))
    for i = 1:length(din)
        switch(din(i))
            case 0 % dcache read miss
                din_full(i) = 0.00000042*1000;
            case 1 % dcache read hit
                din_full(i) = 0.0000001*1000;
            case 2 % dcache write miss
                din_full(i) = 0.00000049*1000;
            case 3 % dcache write hit
                din_full(i) = 0.00000018*1000;
            case 4 % dcache idle
                din_full(i) = 0;
            case 5 % dcache read miss end
                din_full(i) = 0.00000042*1000;
            case 6 % dcache write miss end
                din_full(i) = 0.00000049*1000;
        end
    end
elseif(contains(comp, 'l2cache', 'IgnoreCase', true))
    for i = 1:length(din)
        switch(din(i))
            case 0 % l2 access
                din_full(i) = 0.000000912263*1000;
            case 1 % l2 idle
                din_full(i) = 0;
            case 2 % l2 access end
                din_full(i) = 0.000000912263*1000;
        end
    end
end

end
