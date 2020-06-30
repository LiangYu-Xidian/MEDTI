dim_drug = 1500;
dim_prot = 2500;
dim_imc = 16;
rm_homo = false;
if rm_homo == true;
	interaction = load('../data/mat_drug_protein_remove_homo.txt');
else
%	interaction = load('../data/mat_drug_protein.txt');
	interaction = load('../../../deepNF-master/test_data/mat_drug_protein.txt');
end
%model_200_10_200_10/protein/netsbaseCombineVectNotran.txt;
%drug_feat = load(['../../../deepNF-master/check/drug_zuhezong.txt']);
%prot_feat = load(['../../../deepNF-master/check/protein_zuhezong.txt']);
drug_feat = load(['../../../deepNF-master/test_data/test_results/drug_arch_2_1500_K3_alpha0.9_features.txt']);
prot_feat = load(['../../../deepNF-master/test_data/test_results/protein_arch_2_2500_K3_alpha0.9_features.txt']);
%drug_feat = load(['../../../deepNF-master/data/test_models/drug_d100.txt']);
%prot_feat = load(['../../../deepNF-master/data/test_models/protein_d400.txt']);
%drug_feat = load(['../feature/drug_vector_d',num2str(dim_drug),'.txt']);
%prot_feat = load(['../feature/protein_vector_d',num2str(dim_prot),'.txt']);
nFold = 10;
Nrepeat = 5;

AUROC = zeros(Nrepeat, 1);
AUPRC = zeros(Nrepeat, 1);
re = [];

for p = 1 : Nrepeat
    fprintf('Repetition #%d\n', p);
    [AUROC(p), AUPRC(p), re{p}] = DTINet(p, nFold, interaction, drug_feat, prot_feat, dim_imc);
end
[auc,mi]=max(AUROC);
fprintf('MAX value:Repetition #%d: AUROC=%.6f, AUPR=%.6f\n', mi, AUROC(mi), AUPRC(mi));
prediction=re{1,mi};
dlmwrite('../../../deepNF-master/test_data/prediction.txt', prediction, '\t')
for i = 1 : Nrepeat
	fprintf('Repetition #%d: AUROC=%.6f, AUPR=%.6f\n', i, AUROC(i), AUPRC(i));
end
fprintf('Mean: AUROC=%.6f, AUPR=%.6f\n', mean(AUROC), mean(AUPRC));
