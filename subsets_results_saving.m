function [] = subsets_results_saving(subset_result_filename, total_accuracy,...
    cw_accuracy, avg_total_accuracy,avg_cw_accuracy, confusion_matrices,...
    avg_confusion_matrix, action_names) 

    save (subset_result_filename,...
        'total_accuracy', 'cw_accuracy', 'avg_total_accuracy',...
        'avg_cw_accuracy', 'confusion_matrices', 'avg_confusion_matrix');

    % save confusion matrices as excel files.
    for i = 1:length(confusion_matrices)
        xlswrite([subset_result_filename, '.xlsx'],...
            confusion_matrices{i}, ['confusion_matrix', num2str(i)])
    end
    xlswrite([subset_result_filename, '.xlsx'], action_names, 'text_labels');

    xlswrite([subset_result_filename, '.xlsx'], avg_confusion_matrix);
    xlswrite([subset_result_filename, '.xlsx'], action_names, 'text_labels');

end

