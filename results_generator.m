clear all

data_path = 'inputdir';

%% Make datasets for HBN, HCPD, and PNC
%{
Each dataset should have these fields:
-"mats": 268x268xN connectome data
-"behav_table": restrict this to be a table of all variables you want. size
Nxk, where k is the number of variables. If multiple variables, will
automatically do PCA
-"external_behav_table": needed for PCA
%}

%%% HBN EF
dataset_hbn_ef = struct();
load(fullfile(data_path, 'connectomes_hbn_completecase_1110.mat'));
dataset_hbn_ef.mats = permute(connectomes, [2, 3, 1]);
dataset_hbn_ef.behav_table = readtable(fullfile(data_path, 'hbn_pheno_imaging_zscore.csv'));
dataset_hbn_ef.external_behav_table = readtable(fullfile(data_path, 'hbn_pheno_behavior_zscore.csv'));
dataset_hbn_ef.phenotypes = {'NIH_Card_Sort_Age_Corr_Stnd', 'NIH_Flanker_Age_Corr_Stnd', 'NIH_List_Sort_Age_Corr_Stnd', 'NIH_Processing_Age_Corr_Stnd'};
%dataset_hbn_ef.covar_table = readtable(fullfile(data_path, 'hbn_id_age_only_imaging_only.csv'));
%dataset_hbn_ef.covars = {'age'};

%%% HBN LANG
dataset_hbn_lang = struct();
load(fullfile(data_path, 'connectomes_hbn_completecase_1110.mat'));
dataset_hbn_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_hbn_lang.mats(:, exclude_parcels, :) = 0;
dataset_hbn_lang.behav_table = readtable(fullfile(data_path, 'hbn_pheno_imaging_zscore.csv'));
dataset_hbn_lang.external_behav_table = readtable(fullfile(data_path, 'hbn_pheno_behavior_zscore.csv'));
dataset_hbn_lang.phenotypes = {'CTOPP_BW_S', 'CTOPP_EL_S', 'CTOPP_NR_S', 'CTOPP_RD_S', 'CTOPP_RL_S', 'TOWRE_PDE_Scaled', 'TOWRE_SWE_Scaled', 'TOWRE_Total_Scaled'};
% dataset_hbn_lang.covar_table = readtable(fullfile(data_path, 'hbn_id_age_only_imaging_only.csv'));
% dataset_hbn_lang.covars = {'age'};

%%% PNC EF
dataset_pnc_ef = struct();
load(fullfile(data_path, 'connectomes_pnc_completecase_1291.mat'));
dataset_pnc_ef.mats = permute(connectomes, [2, 3, 1]);
dataset_pnc_ef.behav_table = readtable(fullfile(data_path, 'new_pnc_pheno_imaging_zscore.csv'));
dataset_pnc_ef.external_behav_table = readtable(fullfile(data_path, 'pnc_pheno_behavior_zscore.csv'));
dataset_pnc_ef.phenotypes = {'lnb_tp', 'pcet_acc2', 'pcpt_t_tp'};
% dataset_pnc_ef.covar_table = readtable(fullfile(data_path, 'new_pnc_id_age_only_imaging_only.csv'));
% dataset_pnc_ef.covars = {'age'};

%%% PNC LANG
dataset_pnc_lang = struct();
load(fullfile(data_path, 'connectomes_pnc_completecase_1291.mat'));
dataset_pnc_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_pnc_lang.behav_table = readtable(fullfile(data_path, 'new_pnc_pheno_imaging_zscore.csv'));
dataset_pnc_lang.external_behav_table = readtable(fullfile(data_path, 'pnc_pheno_behavior_zscore.csv'));
dataset_pnc_lang.phenotypes = {'pvrt_cr', 'wrat_cr_std'};
% dataset_pnc_lang.covar_table = readtable(fullfile(data_path, 'new_pnc_id_age_only_imaging_only.csv'));
% dataset_pnc_lang.covars = {'age'};

%%% HCPD EF
dataset_hcpd_ef = struct();
load(fullfile(data_path, 'connectomes_hcpd_completecase_428.mat'));
dataset_hcpd_ef.mats = permute(connectomes, [2, 3, 1]);
dataset_hcpd_ef.behav_table = readtable(fullfile(data_path, 'hcpd_pheno_imaging_zscore.csv'));
dataset_hcpd_ef.phenotypes = {'list_sorting_age_corrected_standard_score', 'picseq_ageadjusted', 'cardsort_ageadjusted', 'flanker_ageadjusted', 'patterncomp_ageadjusted'};
% dataset_hcpd_ef.covar_table = readtable(fullfile(data_path, 'hcpd_id_age_only_imaging_only.csv'));
% dataset_hcpd_ef.covars = {'age'};

%%% HCPD LANG
dataset_hcpd_lang = struct();
load(fullfile(data_path, 'connectomes_hcpd_completecase_428.mat'));
dataset_hcpd_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_hcpd_lang.behav_table = readtable(fullfile(data_path, 'hcpd_pheno_imaging_zscore.csv'));
dataset_hcpd_lang.phenotypes = {'readingtest_age_corrected_std', 'picturevocab_age_corrected_std'};
% dataset_hcpd_lang.covar_table = readtable(fullfile(data_path, 'hcpd_id_age_only_imaging_only.csv'));
% dataset_hcpd_lang.covars = {'age'};



%% WITHIN DATASET
dataset_dictionary = containers.Map({'hbn_ef', 'pnc_ef', 'hcpd_ef', 'hbn_lang', 'pnc_lang', 'hcpd_lang'}, {dataset_hbn_ef, dataset_pnc_ef, dataset_hcpd_ef, dataset_hbn_lang, dataset_pnc_lang, dataset_hcpd_lang});
dataset_names_all = keys(dataset_dictionary);
for my_seed = 1:100
    disp(my_seed)
    for dataset_idx = 1:length(dataset_names_all)  
        dataset_name = dataset_names_all{dataset_idx};
        dataset = dataset_dictionary(dataset_name);
        disp(dataset_name)
        results = train_model(dataset_dictionary(dataset_name),  'model_type', 'ridge', 'seed', my_seed, 'num_folds', 10, 'feat_thresh', 0.05, 'feat_selection', 'p', 'null', 0, "control_covars", 0);
        disp(results.r)
        results.dataset_name = dataset_name;
        r_2_decimals = sprintf('%.2f', results.r);
        save_name = [dataset_name, '_', results.model_type, '_pca_', results.pca_type, '_feat_', results.feat_selection, '_thresh', num2str(results.feat_thresh), '_covary', num2str(results.control_covars), char(dataset.covars{1}), '_null', num2str(results.null), '_r', r_2_decimals, '_seed', num2str(results.seed), '.mat'];
        save(fullfile('savedir',dataset_name, save_name),'results');
    end
end


%% EXTERNAL
dataset_dictionary = containers.Map({'hbn_ef', 'pnc_ef', 'hcpd_ef', 'hbn_lang', 'pnc_lang', 'hcpd_lang'}, {dataset_hbn_ef, dataset_pnc_ef, dataset_hcpd_ef, dataset_hbn_lang, dataset_pnc_lang, dataset_hcpd_lang});
dataset_names_all = keys(dataset_dictionary);
    for dataset_idx = 1:length(dataset_names_all)   
        dataset_name_train = dataset_names_all{dataset_idx};
        for dataset_idx2 = 1:length(dataset_names_all)
            dataset_name_test = dataset_names_all{dataset_idx2};
            disp(dataset_name_train)
            disp(dataset_name_test)
            results = train_model(dataset_dictionary(dataset_name_train), 'external_dataset', dataset_dictionary(dataset_name_test), 'model_type', 'ridge', 'seed', 1, 'feat_thresh', 0.05, 'feat_selection', 'p', 'null', 0, "control_covars", 1);
            disp(results.r)
            results.dataset_name_train = dataset_name_train;
            results.dataset_name_test = dataset_name_test;
            r_2_decimals = sprintf('%.2f', results.r);
            save_name = ['train_' dataset_name_train, '_test_', dataset_name_test, '_', results.model_type, '_pca_', results.pca_type, '_feat_', results.feat_selection, '_thresh', num2str(results.feat_thresh), '_covary', num2str(results.control_covars), char(results.covars{1}), '_null', num2str(results.null), '_r', r_2_decimals, '.mat'];
            save(fullfile('savedir', save_name),'results');
        end
    end

