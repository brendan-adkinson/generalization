function results = train_model(dataset, varargin)
%{
Function to train a ridge CPM or basic CPM model

To use this script for within dataset predictions, you should pass one
dataset

To use this script for external (cross-dataset) predictions, you should
pass two datasets

%}

%% Parse input arguments
p=inputParser;
addRequired(p,'dataset',@isstruct);
addParameter(p,'external_dataset', '', @isstruct);  % external pca table
addParameter(p, 'model_type', 'ridge', @ischar);  % options: ridge, cpm
addParameter(p,'seed',1,@isnumeric);
addParameter(p,'num_folds',10,@isnumeric);
addParameter(p,'feat_thresh', 0.05,@isnumeric);
addParameter(p, 'feat_selection', 'percent', @ischar)  % percent, p
addParameter(p, 'null', 0, @isnumeric)  % percent, p
addParameter(p, 'control_covars', 0, @isnumeric)  % percent, p



parse(p,dataset, varargin{:});

external_dataset = p.Results.external_dataset;
model_type = p.Results.model_type;
seed = p.Results.seed;
num_folds = p.Results.num_folds;
feat_thresh = p.Results.feat_thresh;
feat_selection = p.Results.feat_selection;
null = p.Results.null;
control_covars = p.Results.control_covars == 1;

clearvars p

%% get data

% matrix data (vectorized)
X = mat2edge(dataset.mats)';

% behavioral data
num_behav = length(dataset.phenotypes);
behav_all = dataset.behav_table{:, dataset.phenotypes};
n = length(behav_all);


if control_covars
    if isfield(dataset, 'covars') && isfield(dataset, 'covar_table')
        covars = dataset.covar_table{:, dataset.covars};
        disp('running with covariates')
    else
        disp('control_covars is set to run covariates, but either covars phenotypes or covar_table not specified')
    end
end


% shuffle for null distribution
rng(seed)
if null==1
    shuffle_idx = randperm(length(behav_all));
    behav_all = behav_all(shuffle_idx,:);
    if control_covars && isfield(dataset, 'covars') && isfield(dataset, 'covar_table')
        covars = covars(shuffle_idx,:);
    end
end


% option for no PCA, external PCA, or cross-validated PCA
%
%
if length(dataset.phenotypes)==1
    pca_type = 'none';
    behav = behav_all;
elseif (length(dataset.phenotypes)>1) && (isfield(dataset, 'external_behav_table'))
    % if more than 1 phenotype present and if external pca file exists
    pca_type = 'external';

    % run pca in external (non-imaging) data
    behav_external_pca = dataset.external_behav_table{:, dataset.phenotypes};
    [pca_coeff,score,latent,tsquared,explained,mu] = pca(behav_external_pca);

    % project behavior in main dataset
    behav_pcs = (behav_all-mu)*pca_coeff;
    behav = behav_pcs(:, 1);

elseif (length(dataset.phenotypes)>1) && (~isfield(dataset, 'external_behav_table'))
    % if more than 1 phenotype present but no external pca file


    if ~isstruct(external_dataset)
        % do PCA within cross-validation folds for within-dataset
        % predictions
        pca_type = 'cv';

        % make an empty array to fill in with PC test data (obtained via PCA
        % within cross-validation scheme)
        behav = NaN + zeros(n, 1);

    elseif isstruct(external_dataset)
        % if no behavior-only pca file is provided, you can still do internal
        % PCA when predicting in an external dataset

        pca_type = 'internal';

        % run pca in external (non-imaging) data
        [pca_coeff,score,latent,tsquared,explained,mu] = pca(behav_all);

        % project behavior in main dataset
        behav_pcs = (behav_all-mu)*pca_coeff;
        behav = behav_pcs(:, 1);
    end
end


rng(seed)

%% Within-dataset
if isempty(external_dataset)

    % Select cross-validation splits
    cv_idx = cv_indices(n, num_folds);
    coef_all = zeros(size(X, 2), num_folds);
    coef0_all = zeros(num_folds, 1);
    y_predict = zeros(n, 1);
    for k = 1:num_folds

        train_idx = find(cv_idx~=k);
        test_idx = find(cv_idx==k);

        X_train = X(train_idx, :);
        X_test = X(test_idx, :);
        

        if strcmp(pca_type, 'cv')
            % run pca in training data
            [pca_coeff,score,latent,tsquared,explained,mu] = pca(behav_all(train_idx, :));

            % project behavior in training data
            behav_pcs = (behav_all(train_idx, :)-mu)*pca_coeff;
            behav_train = behav_pcs(:, 1);

            % project behavior in test data
            behav_pcs = (behav_all(test_idx, :)-mu)*pca_coeff;
            behav_test = behav_pcs(:, 1);
            behav(test_idx) = behav_test;  % save the test cross-validated PC for later
        else
            behav_train = behav(train_idx);
            behav_test = behav(test_idx);
        end


        % Step 1 of feature selection: correlation
        if control_covars
            covars_train = covars(train_idx);
            % select features with partial correlation
            [edge_corr, edge_p] = partialcorr(X_train, behav_train, covars_train); % was train.control, this.control
        else
            % select features with standar correlation
            [edge_corr, edge_p] = corr(X_train, behav_train);
        end

        % Step 2 of feature selection: thresholding
        if strcmp(feat_selection, 'p')  % selecting by p value
            p_thresh = feat_thresh;
        elseif strcmp(feat_selection, 'percent')  % selecting percentage of features
            p_thresh = prctile(edge_p, 100*feat_thresh);
        end
        feat_loc = find(edge_p<p_thresh);

        % model fitting
        if strcmp(model_type, 'ridge')
            [fit_coef, fit_info] = lasso(X_train(:, feat_loc), behav_train, 'Alpha',1e-6, 'CV', num_folds);  % low alpha for ridge
            idxLambda1SE = fit_info.Index1SE;
            coef = fit_coef(:,idxLambda1SE);
            coef0 = fit_info.Intercept(idxLambda1SE);
            coef_all(feat_loc, k) = coef;
            coef0_all(k) = coef0;

            % prediction
            y_predict(test_idx) = X_test(:, feat_loc)*coef+coef0;
        elseif strcmp(model_type, 'cpm')  % TODO: we may need to think about what you want to save with CPM
            % get positive and negative networks, and summarize into a
            % single feature
            pos_network = find(feat_loc(edge_corr(feat_loc)>0));
            neg_network = find(feat_loc(edge_corr(feat_loc)<0));        
            X_train_summary = sum(X_train(:, pos_network), 2) - sum(X_train(:, neg_network), 2);

            % fit model
            mdl = robustfit(X_train_summary,behav_train);  

            % predict
            X_test_summary = sum(X_test(:, pos_network), 2) - sum(X_test(:, neg_network), 2);
            y_predict(test_idx) = mdl(1) + mdl(2)*X_test_summary;

        end


    end

    % store results
    results.y_true = behav;
    results.y_predict = y_predict;
    results.cv_idx = cv_idx;
    results.lambda = fit_info.Lambda1SE;
    results.coef_mean = mean(coef_all, 2);
    results.coef0_mean = mean(coef0_all, 2);
    results.r = corr(results.y_true,results.y_predict);
    results.pca_type = pca_type;
    results.phenotypes = dataset.phenotypes;
    results.model_type = model_type;
    results.seed = seed;
    results.num_folds = num_folds;
    results.feat_thresh = feat_thresh;
    results.feat_selection = feat_selection;
    results.null = null; % 1 is for null
    results.control_covars = control_covars; 
    if control_covars
        results.covars = dataset.covars;
    else
        results.covars = 'none';
    end






%% Across datasets
else

    % get second (external) dataset X data 
    X_external = mat2edge(external_dataset.mats)';
    behav_all_external = external_dataset.behav_table{:, external_dataset.phenotypes};

    % for second (external) y data, depends if PCA is needed
    if length(external_dataset.phenotypes)==1
        behav_external = behav_all_external;
    elseif (length(external_dataset.phenotypes)>1) && (isfield(external_dataset, 'external_behav_table'))
        % if more than 1 phenotype present and if external pca file exists
        pca_type = 'external';

        % run pca in external (non-imaging) data
        behav_external_external_pca = external_dataset.external_behav_table{:, external_dataset.phenotypes}; % added "external_" before both "datasets:
        [pca_coeff,score,latent,tsquared,explained,mu] = pca(behav_external_external_pca);

        % project behavior in main dataset
        behav_pcs_external = (behav_all_external-mu)*pca_coeff;
        behav_external = behav_pcs_external(:, 1);

    elseif (length(dataset.phenotypes)>1) && (~isfield(dataset, 'external_behav_table'))
        pca_type = 'internal';

        % run pca in external dataset
        behav_external_external_pca = dataset.external_behav_table{:, dataset.phenotypes};
        [pca_coeff,score,latent,tsquared,explained,mu] = pca(behav_external_external_pca);

        % project behavior in main dataset
        behav_pcs_external = (behav_all_external-mu)*pca_coeff;
        behav_external = behav_pcs_external(:, 1);
    end



    % Step 1 of feature selection: correlation
    if control_covars
        % select features with partial correlation
        [edge_corr, edge_p] = partialcorr(X, behav, covars); % was train.control, this.control
    else
        % select features with standar correlation
        [edge_corr, edge_p] = corr(X, behav);
    end

    % Step 2 of feature selection: thresholding
    if strcmp(feat_selection, 'p')  % selecting by p value
        p_thresh = feat_thresh;
    elseif strcmp(feat_selection, 'percent')  % selecting percentage of features
        p_thresh = prctile(edge_p, 100*feat_thresh);
    end
    feat_loc = find(edge_p<p_thresh);

    % model training
    [fit_coef, fit_info] = lasso(X(:, feat_loc), behav, 'Alpha',1e-6, 'CV', num_folds);  % changed X to X(:, feat_loc)
    idxLambda1SE = fit_info.Index1SE;
    coef = fit_coef(:,idxLambda1SE);
    coef0 = fit_info.Intercept(idxLambda1SE);

    % prediction
    y_predict = X_external(:, feat_loc)*coef+coef0;

    % store results
    results.y_train = behav;
    results.y_external_true = behav_external;
    results.y_external_predict = y_predict;
    results.coef = coef;
    results.coef0 = coef0;
    results.lambda = fit_info.Lambda1SE;
    results.coef = coef;
    results.coef0 = coef0;
    results.r = corr(results.y_external_true,results.y_external_predict);
    results.pca_type = pca_type;
    results.phenotypes = dataset.phenotypes; % which variables were used
    results.model_type = model_type;
    results.seed = seed;
    results.feat_thresh = feat_thresh;
    results.feat_selection = feat_selection;
    results.null = null; % 1 is for null
    results.control_covars = control_covars; % whether to control for covariates
    if control_covars
        results.covars = dataset.covars; % which variables were used as covariates
    else
        results.covars = 'none';
    end
end

