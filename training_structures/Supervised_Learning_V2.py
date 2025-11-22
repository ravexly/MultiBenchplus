"""Implements supervised learning training procedures."""
import time
import sys
import torch
from torch import nn
from tqdm import tqdm

from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from transformers.tokenization_utils_base import BatchEncoding
# from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot

softmax = nn.Softmax()

# Global lists to track metrics across runs
allacc = []
allloss = []
allf1 = []


class MMDL(nn.Module):
    """
    Implements the Multi-Modal Deep Learning (MMDL) classifier.
    """

    def __init__(self, encoders, fusion, head, has_padding=False, tokenizer=None, fusion_type='feature'):
        """
        Instantiate MMDL Module.

        Args:
            encoders (list[nn.Module]): A list of encoders, one for each modality.
            fusion (nn.Module): The fusion module.
            head (nn.Module or list[nn.Module]): A classifier module or a list of them for decision-level fusion.
            has_padding (bool, optional): Whether the input has padding. Defaults to False.
            tokenizer (optional): A tokenizer for text processing. Defaults to None.
            fusion_type (str, optional): The type of fusion, 'feature' or 'decision'. Defaults to 'feature'.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = nn.ModuleList(head) if fusion_type != 'feature' else head
        self.has_padding = has_padding
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_type = fusion_type
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """
        Apply MMDL to the input layer.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        outs = []
        if self.has_padding:
            # Handle padded data (element-wise)
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            # Handle non-padded data (batch-wise)
            if self.tokenizer is not None:
                # Special handling for text modality
                # print(type(inputs[1]))
                # Check if inputs[1] is raw text (e.g., list of strings) or already tokenized (dict)
                if isinstance(inputs[1], BatchEncoding):
                    # print(inputs[1])
                    # If it's a dict, it's already tokenized. Use it directly.
                    encoded_inputs = inputs[1]
                else:
                    # If it's not a dict, assume it's raw text and encode it.
                    encoded_inputs = self.tokenizer.batch_encode_plus(
                        inputs[1],
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    )
                
                # Move tensors to the correct device
                # Using .get() is safer in case a key is missing
                txt = encoded_inputs['input_ids'].to(self.device)
                mask = encoded_inputs['attention_mask'].to(self.device)
                segment_tensor = encoded_inputs.get('token_type_ids') # .get() is safer than ['...']
                segment = segment_tensor.to(self.device) if segment_tensor is not None else None
                
                # Process the first two modalities
                outs.append(self.encoders[0](inputs[0]).squeeze(0))
                
                # Handle the case where segment might be None
                if segment is not None:
                    outs.append(self.encoders[1](txt, mask, segment))
                else:
                    # Assumes the encoder can be called without token_type_ids
                    outs.append(self.encoders[1](txt, mask))

                # Process remaining modalities (starting from the 3rd)
                for i in range(2, len(inputs)):
                    outs.append(self.encoders[i](inputs[i]))
            else:
                # General handling for cases without a tokenizer
                for i in range(len(inputs)):
                    outs.append(self.encoders[i](inputs[i]).squeeze(0))

        if self.fusion_type == 'feature':
            # Feature-level fusion
            # print(outs)
            if self.has_padding:
                out = self.fuse([i[0] for i in outs]) if not isinstance(outs[0], torch.Tensor) else self.fuse(outs)
            else:
                out = self.fuse(outs)
            
            if isinstance(out, tuple):
                out = out[0]
            
            return self.head(out)

        elif self.fusion_type == 'decision':
            # Decision-level fusion
            if self.head is None:
                raise ValueError("Heads must be provided for decision-level fusion.")
            
            logits = [head(out) for head, out in zip(self.head, outs)]
            logits = self.fuse(logits)
            return logits
        else:
            raise ValueError("Invalid fusion type. Choose 'feature' or 'decision'.")


def deal_with_objective(objective, pred, truth, args):
    """
    Alter inputs depending on the objective function to handle different arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
        
    if isinstance(objective, nn.CrossEntropyLoss):
        if len(truth.size()) == len(pred.size()):
            truth = truth.squeeze(len(pred.size()) - 1)
        return objective(pred, truth.long().to(device))
    
    elif isinstance(objective, (nn.MSELoss, nn.BCEWithLogitsLoss, nn.L1Loss)):
        return objective(pred, truth.float().to(device))
    
    else:
        # Custom objective function
        return objective(pred, truth, args)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs,
        additional_optimizing_modules=[], is_packed=False, early_stop=False,
        task="classification", optimtype=torch.optim.RMSprop, lr=0.001,
        weight_decay=0.0, objective=nn.CrossEntropyLoss(), auprc=False,
        save='best.pt', validtime=False, objective_args_dict=None,
        input_to_float=True, clip_val=8, track_complexity=True,
        tokenizer=None, freeze_encoders=False, fusion_type='feature'):
    """
    Handles a simple supervised training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Freeze encoder parameters
    if freeze_encoders:
        for enc in encoders:
            for param in enc.parameters():
                param.requires_grad = False

    model = MMDL(encoders, fusion, head, has_padding=is_packed, tokenizer=tokenizer, fusion_type=fusion_type).to(device)

    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend([p for p in m.parameters() if p.requires_grad])
        
        op = optimtype([p for p in model.parameters() if p.requires_grad] + additional_params, lr=lr, weight_decay=weight_decay)
        
        best_val_loss = 10000
        best_acc = 0
        best_f1 = 0
        patience = 0

        def _processinput(inp):
            if isinstance(inp, torch.Tensor):
                return inp.float().to(device) if input_to_float else inp.to(device)
            # If not a tensor, it might be text or other info; return the original value
            return inp

        for epoch in range(total_epochs):
            total_loss = 0.0
            total_samples = 0
            model.train()

            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{total_epochs} (Train)", unit="batch") as train_bar:
                for j in train_bar:
                    # print(j)
                    j[-1] = j[-1].to(device)
                    op.zero_grad()
                    
                    if is_packed:
                        out = model([[_processinput(i) for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i) for i in j[:-1]])
                    
                    if objective_args_dict is not None:
                        objective_args_dict.update({
                            'reps': model.reps,
                            'fused': model.fuseout,
                            'inputs': j[:-1],
                            'training': True,
                            'model': model,
                            'epoch': epoch
                        })
                    
                    loss = deal_with_objective(objective, out, j[-1], objective_args_dict)
                    total_loss += loss.item() * len(j[-1])
                    total_samples += len(j[-1])
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    op.step()
                    
                    train_bar.set_postfix(loss=total_loss / total_samples)

            print(f"Epoch {epoch + 1} train loss: {total_loss / total_samples:.4f}")

            # Validation phase
            valid_start_time = time.time()
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                preds, truths, pts = [], [], []

                with tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{total_epochs} (Valid)", unit="batch") as valid_bar:
                    for j in valid_bar:
                        j[-1] = j[-1].to(device)
                        
                        if is_packed:
                            out = model([[_processinput(i) for i in j[0]], j[1]])
                        else:
                            out = model([_processinput(i) for i in j[:-1]])
                        
                        if objective_args_dict is not None:
                            objective_args_dict.update({
                                'reps': model.reps,
                                'fused': model.fuseout,
                                'inputs': j[:-1],
                                'training': False
                            })
                        
                        loss = deal_with_objective(objective, out, j[-1], objective_args_dict)
                        total_loss += loss.item() * len(j[-1])
                        
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        
                        if task == "classification":
                            preds.append(torch.argmax(out, 1))
                        elif task == "multilabel":
                            preds.append(torch.sigmoid(out).round())
                        
                        truths.append(j[-1])
                        
                        if auprc:
                            sm = softmax(out)
                            pts.extend([(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))])
                            
                        valid_bar.set_postfix(loss=total_loss / total_samples)

                preds = torch.cat(preds, 0) if preds else torch.empty(0)
                truths = torch.cat(truths, 0)
                total_samples = truths.shape[0]
                val_loss = total_loss / total_samples
                
                if task == "classification":
                    acc = accuracy(truths, preds)
                    print(f"Epoch {epoch + 1} valid loss: {val_loss:.4f}, acc: {acc:.4f}")
                    if acc > best_acc:
                        patience = 0
                        best_acc = acc
                        print("Saving Best Model")
                        torch.save(model, save)
                    else:
                        patience += 1
                elif task == "multilabel":
                    f1_micro = f1_score(truths, preds, average="micro")
                    f1_macro = f1_score(truths, preds, average="macro")
                    print(f"Epoch {epoch + 1} valid loss: {val_loss:.4f}, f1_micro: {f1_micro:.4f}, f1_macro: {f1_macro:.4f}")
                    if f1_macro > best_f1:
                        patience = 0
                        best_f1 = f1_macro
                        print("Saving Best Model")
                        torch.save(model, save)
                    else:
                        patience += 1
                elif task == "regression":
                    print(f"Epoch {epoch + 1} valid loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        patience = 0
                        best_val_loss = val_loss
                        print("Saving Best Model")
                        torch.save(model, save)
                    else:
                        patience += 1

                if early_stop and patience > 7:
                    print("Early stopping triggered.")
                    break
                
                if auprc:
                    print(f"AUPRC: {AUPRC(pts):.4f}")
                
                if validtime:
                    print(f"Valid time: {time.time() - valid_start_time:.2f}s")
        
        if task == "classification":
            allacc.append(best_acc)
            return best_acc
        elif task == "regression":
            allloss.append(best_val_loss)
            return best_val_loss
        elif task == 'multilabel':
            allf1.append(best_f1)
            return best_f1

    if track_complexity:
        return all_in_one_train(_trainprocess, [model] + additional_optimizing_modules)
    else:
        return _trainprocess()


def single_test(model, test_dataloader, is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    """
    Run a single test loop for a model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _processinput(inp):
        if isinstance(inp, torch.Tensor):
            return inp.float().to(device) if input_to_float else inp.to(device)
        return inp

    with torch.no_grad():
        total_loss = 0.0
        preds, truths, pts = [], [], []

        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i) for i in j[0]], j[1]])
            else:
                out = model([_processinput(i) for i in j[:-1]])

            if isinstance(out, (list, tuple)):
                out = out[0]
            
            loss = deal_with_objective(criterion, out, j[-1], None)
            total_loss += loss.item() * len(j[-1])

            if task == "classification":
                preds.append(torch.argmax(out, 1))
            elif task == "multilabel":
                preds.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                oute = out.cpu().numpy().tolist()
                prede = [1 if i[0] > 0 else -1 if i[0] < 0 else 0 for i in oute]
                preds.append(torch.LongTensor(prede))
            
            truths.append(j[-1])

            if auprc:
                sm = softmax(out)
                pts.extend([(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))])
        
        preds = torch.cat(preds, 0) if preds else torch.empty(0)
        truths = torch.cat(truths, 0)
        total_samples = truths.shape[0]
        test_loss = total_loss / total_samples

        if auprc:
            print(f"AUPRC: {AUPRC(pts):.4f}")
        
        if task == "classification":
            acc = accuracy(truths, preds)
            print(f"Accuracy: {acc:.4f}")
            return acc
        elif task == "multilabel":
            f1_micro = f1_score(truths, preds, average="micro")
            f1_macro = f1_score(truths, preds, average="macro")
            print(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")
            return {'micro': f1_micro, 'macro': f1_macro}
        elif task == "regression":
            print(f"MSE: {test_loss:.4f}")
            return test_loss
        elif task == "posneg-classification":
            accs = eval_affect(truths, preds)
            print(f"Accuracy (excluding zeros): {accs:.4f}")
            return {'Accuracy': accs}


def test(model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False):
    """
    Handles getting test results and robustness evaluation.
    """
    if no_robust:
        def _testprocess():
            return single_test(model, test_dataloaders_all, is_packed, criterion, task, auprc, input_to_float)
        return all_in_one_test(_testprocess, [model])

    # Standard test on clean data
    def _testprocess_clean():
        clean_dataloader = test_dataloaders_all[list(test_dataloaders_all.keys())[0]][0]
        return single_test(model, clean_dataloader, is_packed, criterion, task, auprc, input_to_float)
    
    all_in_one_test(_testprocess_clean, [model])
    
    # Robustness testing
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print(f"Testing on noisy data ({noisy_modality})...")
        robustness_curve = {}
        
        for test_dataloader in tqdm(test_dataloaders, desc=f"Noise on {noisy_modality}"):
            single_test_result = single_test(model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
            
            if isinstance(single_test_result, dict):
                for k, v in single_test_result.items():
                    robustness_curve.setdefault(k, []).append(v)
            else:
                 robustness_curve.setdefault('metric', []).append(single_test_result)
        
        for measure, result_curve in robustness_curve.items():
            robustness_key = f'{dataset} {noisy_modality}'
            rel_robust = relative_robustness(result_curve, robustness_key)
            eff_robust = effective_robustness(result_curve, robustness_key)
            
            print(f"Relative Robustness ({noisy_modality}, {measure}): {rel_robust:.4f}")
            print(f"Effective Robustness ({noisy_modality}, {measure}): {eff_robust:.4f}")
            
            fig_name = f'{method_name}-{robustness_key}-{noisy_modality}-{measure}'.replace(" ", "_")
            single_plot(result_curve, robustness_key, xlabel='Noise Level', ylabel=measure, fig_name=fig_name, method=method_name)
            print(f"Plot saved as {fig_name}.png")