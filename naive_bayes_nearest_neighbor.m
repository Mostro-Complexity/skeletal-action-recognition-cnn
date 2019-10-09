function [total_accuracy, class_wise_accuracy, confusion_matrix]...
    = naive_bayes_nearest_neighbor(features_train, features_test, tr_labels, te_labels, ~)

    n_tr_samples = length(features_train);
    n_te_samples = length(features_test);
    unique_classes = unique(tr_labels);
    n_classes = length(unique_classes);


    predicted_ind = zeros(n_te_samples, 1);

    for video_ind = 1:n_te_samples
        video = features_test{video_ind};

        each_frame_dist = zeros(n_classes, 1);
        for c = 1:n_classes

            videos_in_c = features_train(tr_labels == unique_classes(c));
            frames_in_c = cell2mat(videos_in_c');
            nn_ind = knnsearch(frames_in_c', video');

            % frames:norm(di-NNc_di)       norm:L1 norm
            each_frame_norms = sum(abs(video - frames_in_c(:, nn_ind)), 1);

            each_frame_dist(c) = sum(each_frame_norms.^2);

        end

        [~, predicted_ind(video_ind)] = min(each_frame_dist);
    end
    
    pr_labels = unique_classes(predicted_ind);
    
    % evaluation
    class_wise_accuracy = zeros(n_classes, 1);    
    confusion_matrix = zeros(n_classes, n_classes);    
    for i = 1:n_classes     % arranged by order
        temp = find(te_labels == unique_classes(i));
        class_wise_accuracy(i) =...
            length(find(pr_labels(temp) == unique_classes(i))) / length(temp);
        
         confusion_matrix(i, :) = hist(pr_labels(temp),...
             unique_classes) / length(temp);
    end
    
    total_accuracy = length(find(te_labels == pr_labels))...
        / n_te_samples;

end

