function [] = nbnn_classification_with_subsets(root_dir, subject_labels,...
    action_labels, tr_subjects, te_subjects, action_sets, action_names)
  
    results_dir = [root_dir, '/nbnn_modeling_results'];


    n_tr_te_splits = size(tr_subjects, 1);    

    n_action_sets = length(action_sets);
  
    C_val = 1;

    loadname = 'features';

    for set = 1:n_action_sets

        actions = unique(action_sets{set});
        n_classes = length(actions);      

        total_accuracy = zeros(n_tr_te_splits, 1);        
        cw_accuracy = zeros(n_tr_te_splits, n_classes);
        confusion_matrices = cell(n_tr_te_splits, 1);

        action_ind = ismember(action_labels, actions);  
        for i = 1:n_tr_te_splits         
            tr_subject_ind = ismember(subject_labels, tr_subjects(i,:));
            te_subject_ind = ismember(subject_labels, te_subjects(i,:));        

            tr_ind = (action_ind & tr_subject_ind);
            te_ind = (action_ind & te_subject_ind);                

            tr_labels = action_labels(tr_ind);
            te_labels = action_labels(te_ind);

            data = load ([root_dir, '/', loadname], loadname);

            features = data.(loadname);
            features_train = features(tr_ind);
            features_test = features(te_ind);

            [total_accuracy(i), cw_accuracy(i,:), confusion_matrices{i}] =...
                naive_bayes_nearest_neighbor(features_train,...
                features_test, tr_labels, te_labels, C_val);

        end

        avg_total_accuracy = mean(total_accuracy);               
        avg_cw_accuracy = mean(cw_accuracy);

        avg_confusion_matrix = zeros(size(confusion_matrices{1}));
        for j = 1:length(confusion_matrices)
            avg_confusion_matrix = avg_confusion_matrix + confusion_matrices{j};
        end
        avg_confusion_matrix = avg_confusion_matrix / length(confusion_matrices);

        subsets_results_saving(...
            [results_dir, '/classification_results_as', num2str(set)],...
            total_accuracy, cw_accuracy, avg_total_accuracy, avg_cw_accuracy,...
            confusion_matrices, avg_confusion_matrix, action_names(actions));
        
    end
end
