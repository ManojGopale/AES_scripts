% This is the main file that runs power trace generation
% for each configuration
%%Mfunction [power] = generateTrace(qResult_folders, outputPath)
function [power] = generateTrace(qResult_folders, outputPath, startKey, endKey)
    traceType = ["CPU", "Icache", "Dcache", "L2cache"];
		% Commenting to parallelize the matlab runs too
    %for key=0:255 %key ranges from 0-255 in our experiments
    for key=startKey:endKey %key ranges from 0-255 in our experiments
        for traceIndex = 1:length(traceType)
            if(contains(traceType{traceIndex}, "cpu", 'IgnoreCase', true))
                %Call CPU to get ix and iy
                [ix, iy, inter_dout] = BinRead(1,key, qResult_folders, traceType{traceIndex});
                % Since this is the first loop for every key, initialize
                % power to 0 here with the size for inter_dout
                power = zeros(size(inter_dout));
            else
                [~,~, inter_dout] = BinRead(1,key, qResult_folders, traceType{traceIndex}, ix, iy);
            end
            power = power + inter_dout; %Adding inter dout's before going to next trace type   
        end
        
        % Save file to value*.mat after every key
        % 30,000x5361 takes 42Mb, Saving file in .mat format
        fileName = sprintf("%s/value%d.mat", outputPath, key);
        save(fileName, 'power');
				fprintf('Saving file to\n%s', fileName);

        %OR I can save it to csv file directly and then use it in python
        %csvFileName = sprintf("%s/value%d.csv", outputPath, key);
        
        % writematrix is intoduced in 2019 version
        %writematrix(power, csvFileName);
        
        % For 2018 version, can be used directly in pandas
        % 30,000x5361 takes 1.3Gb
        %Mcsvwrite(csvFileName, power);
    end
end
