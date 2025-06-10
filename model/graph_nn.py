import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self,task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features=torch.cat([aligned_task_features,aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask=F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        edge_predict = F.sigmoid(
            (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))
        return edge_predict

class form_data:

    def __init__(self,device):
        self.device = device

    def formulation(self,task_id,query_feature,llm_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask):

        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id=torch.tensor(task_id, dtype=torch.float).to(self.device)
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node=[(i+1 + org_node[-1]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1,1).to(self.device)
        combined_edge=torch.tensor(combined_edge, dtype=torch.float).reshape(-1,2).to(self.device)
        combined_edge=torch.cat((edge_weight, combined_edge), dim=-1)
        data = Data(task_id=task_id,query_features=query_features, llm_features=llm_features, edge_index=edge_index,
                        edge_attr=edge_weight,query_indices=query_indices, llm_indices=llm_indices,label=torch.tensor(label, dtype=torch.float).to(self.device),
                        edge_mask=edge_mask,combined_edge=combined_edge,
                    train_mask=train_mask,valide_mask=valide_mask,test_mask=test_mask)

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim,hidden_features_size,in_edges_size,wandb,config,device):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,in_edges=in_edges_size).to(device)
        self.wandb = wandb
        self.config = config
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = torch.nn.BCELoss()

    def train_validate(self,data,data_validate,data_for_test):

        best_f1=-1
        self.save_path= self.config['model_path']
        self.num_edges = len(data.edge_attr)
        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)
        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean=0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone()
                mask = mask.bool()
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool)
                mask = torch.where(mask & random_mask, torch.tensor(False, dtype=torch.bool), mask)
                mask = mask.bool()
                edge_can_see = torch.logical_and(~mask, self.train_mask)
                self.optimizer.zero_grad()
                predicted_edges= self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                                            edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean+=loss
            loss_mean=loss_mean/self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()

            self.model.eval()
            mask_validate = torch.tensor(data_validate.edge_mask, dtype=torch.bool)
            edge_can_see = self.train_mask
            with torch.no_grad():
                predicted_edges_validate = self.model(task_id=data_validate.task_id,query_features=data_validate.query_features,
                                                                            llm_features=data_validate.llm_features,
                                                                            edge_index=data_validate.edge_index,
                                                                            edge_mask=mask_validate,edge_can_see=edge_can_see, edge_weight=data_validate.combined_edge)
                observe_edge= predicted_edges_validate.reshape(-1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, 1)
                value_validate=data_validate.edge_attr[mask_validate].reshape(-1, self.config['llm_num'])
                label_idx = torch.argmax(value_validate, 1)
                correct = (observe_idx == label_idx).sum().item()
                total = label_idx.size(0)
                validate_accuracy = correct / total
                observe_idx_ = observe_idx.cpu().numpy()
                label_idx_ = label_idx.cpu().numpy()
                # calculate macro F1 score
                f1 = f1_score(label_idx_, observe_idx_, average='macro')
                loss_validate = self.criterion(predicted_edges_validate.reshape(-1), data_validate.label[mask_validate].reshape(-1))

                if f1>best_f1:
                    best_f1 = f1
                    torch.save(self.model.state_dict(), self.save_path)
                test_result,test_loss=self.test(data_for_test,self.config['model_path'])
                self.wandb.log({"train_loss":loss_mean,"validate_loss": loss_validate,"test_loss":test_loss, "validate_accuracy": validate_accuracy,"validate_f1": f1, "test_result": test_result})

    def test(self,data,model_path):
        # self.model.load_state_dict(model_path)
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                             edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        print("result_predict:", result, "result_golden:",result_golden)

        return result,loss_test