%% nbnn model
% 'histograms_of_3D_joint_locations ',
feature_types = {'absolute_joint_positions'};

datasets = {'MSRAction3D'};

tic
for i = 1:length(datasets)
    for j = 1:length(feature_types)
        naive_bayes_nearest_neighbor_modeling(i, j, feature_types, datasets);
    end
end
toc