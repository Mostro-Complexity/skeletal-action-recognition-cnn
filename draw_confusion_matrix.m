function draw_confusion_matrix(mat, tick)
    n_classes = length(tick);
    imagesc(1:n_classes, 1:n_classes,mat);            %# in color

    colormap(flipud(gray));  %# for gray; black for large value.

    textStrings = num2str(mat(:),'%0.1f');  
    textStrings = strtrim(cellstr(textStrings)); 

    [x,y] = meshgrid(1:n_classes); 

    hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
    midValue = mean(get(gca,'CLim')); 
    
    textColors = repmat(mat(:) > midValue,1,3); 
    textColors(mat(:) < 5, :) = 1;   

    set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
    set(gca,'xticklabel',tick,'XAxisLocation','bottom');
    set(gca, 'XTick', 1:n_classes, 'YTick', 1:n_classes);
    set(gca,'yticklabel',tick);
    set(gca,'TickLength',[0.005, 0.001]);
    set(gcf,'position',[500,300,808,420]);
    xtickangle(50)

end 

 

 
