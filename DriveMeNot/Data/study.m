% DS=["trace4.csv"];
% size(DS)
% filename="trace4.csv";
% % filenamew = strcat('WiFi_', filename);
% % filenamewithpath = sprintf('./Data/%s', filenamew);
% % obj.filename=filename;
% % 
% % fprintf('Processing trace: %s\n', filenamew);
% % filenamewithpath = sprintf('./Data/%s', filenamew);
% % Tw = readtable(filenamewithpath);
% % 
% % Dw.info = table2array(Tw(:, [1, 2, 5, 9]));
% % Dw.bssid = cell2mat(Tw.('BSSID'));                
% % obj.Dw = Dw
% 
% % obj.Dw;
% a=[1,2;3,4,5]
% b=[a;[5,6;7,8]]
% x = [0 2 9 2 5 8 7 3 1 9 4 3 5 8 10 0 1 2 9 5 10];
% count=hist(x,3)
% %% run
% a=[1,2;3,4];
% b=[a;[5,6;7,8]];
% x = [0 2 9 2 5 8 7 3 1 9 4 3 5 8 10 0 1 2 9 5 10];
% count=hist(x,3)




% % Sample data
% di = [2.3, 2.9, 3.1, 2.8, 2.7, 3.0, 3.2, 3.1, 2.5, 2.8];
% 
% % Fit all available distributions to the data
% [D, PD] = allfitdist(di, 'PDF');
% 
% % Display the fitted distributions and their parameters
% disp('Fitted Distributions and Parameters:');
% for i = 1:length(D)
%     fprintf('Distribution: %s\n', D(i).DistName);
%     disp(D(i).Params);
% end
% 
% % Plot the PDFs of the fitted distributions
% figure;
% hold on;
% x_values = linspace(min(di), max(di), 100);
% for i = 1:length(PD)
%     y_values = pdf(PD{i}, x_values);
%     plot(x_values, y_values, 'DisplayName', D(i).DistName);
% end
% legend show;
% title('Fitted Distributions');
% xlabel('Data');
% ylabel('Probability Density');
% hold off;



% 
% % Step 1: Create a normal distribution with mean 0 and standard deviation 1
% pd = makedist('Normal', 'mu', 0, 'sigma', 1);
% 
% % Step 2: Generate 10 random numbers from this distribution
% randomNumbers = random(pd, 10, 1);
% disp('Random Numbers:');
% disp(randomNumbers);
% 
% % Step 3: Compute the mean and standard deviation of the distribution
% meanVal = mean(pd);
% stdVal = std(pd);
% disp(['Mean: ', num2str(meanVal)]);
% disp(['Standard Deviation: ', num2str(stdVal)]);
% 
% % Step 4: Evaluate the PDF at x = 1
% pdfValue = pdf(pd, 1);
% disp(['PDF at x=1: ', num2str(pdfValue)]);
% 
% % Step 5: Evaluate the CDF at x = 1
% cdfValue = cdf(pd, 1);
% disp(['CDF at x=1: ', num2str(cdfValue)]);
% 
% % Step 6: Plot the PDF of the distribution
% x = -3:0.1:3;
% y = pdf(pd, x);
% figure;
% plot(x, y);
% title('PDF of Normal Distribution');
% xlabel('x');
% ylabel('Probability Density');


for i = [2, 5:5:100]
    i
end