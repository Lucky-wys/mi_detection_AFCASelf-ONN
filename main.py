from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


from Data.data_utils_2 import gen_data,get_batch
from Models.self_onn_fcanet_ada3 import self_onn_fcanet_ada3
from Models.self_onn_fcanet import self_onn_fcanet
from Models.self_onn import self_onn
from Models.CNN import CNN
from Models.self_onn_fcanet_ada3_xavier import self_onn_fcanet_ada3_xavier
def get_current_datetime():
        # 获取当前的日期和时间
        current_time = datetime.datetime.now()
        
        # 将日期和时间拼接成字符串
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        
        # 返回拼接后的时间字符串
        return formatted_time

for i in range(3):
    batch_size = 10
    data_name = "ptbdb0402"
    exp_name = "self-onn_Fcanet_ada_16-8_xavier_32_19"

    seed_num = 42
    riqi = get_current_datetime()
    result_path = f"result/{data_name}/{seed_num}/{exp_name}/{riqi}"

    train_data,val_data,test_data = gen_data(result_path,seed_num,chns = ['vz', 'v6'])

    results_path = os.path.join(result_path,'results')

    model = self_onn_fcanet_ada3_xavier().to('cuda:0')
    # model.cuda()

    model = nn.DataParallel(model, device_ids=[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir=results_path)


    num_iters = 35000
    batch_size = 10
    acc_values = []
    acc_values_train = []

    loss_values = []
    loss_values_train = []
    # 初始化最高准确率为 0
    best_acc = 0
    best_iter = 0

    results_file_path_plot = os.path.join(result_path, 'plot')
    os.makedirs(results_file_path_plot, exist_ok=True)
    results_file_path = os.path.join(results_file_path_plot, 'training_results.txt')


    best_loss = float('inf')  # 设置初始最低损失为无穷大
    saved_models = []  # 用于存储保存的模型路径
    saved_models_test = []  # 用于存储测试集上最佳模型路径
    max_saved_models = 10  # 最多保存的模型数量
    best_test_acc = 0

    # 修改后的训练过程
    for iters in range(num_iters):
        batch_x, batch_y = get_batch(batch_size,train_data,val_data,test_data,window_size=10000, split='train')
        batch_x, batch_y = batch_x.to('cuda:0'), batch_y.to('cuda:0')

        y_pred = model(batch_x)

        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        if iters % 100 == 0 and iters != 0:

            writer.add_scalar('Loss/train', loss, iters)
            print('Loss/train=', loss.cpu().detach().numpy())

            with torch.no_grad():

                # validation
                iterations = 100
                avg_acc = 0
                total_val_loss = 0  # 用于累计验证损失   +++

                for _ in range(iterations):
                    batch_x, batch_y = get_batch(batch_size,train_data,val_data,test_data,window_size=10000, split='val')
                    
                    cleaned = model(batch_x)

                    # 计算验证损失
                    val_loss = criterion(cleaned, batch_y)   # ++
                    total_val_loss += val_loss.item()   # ++

                    count = 0
                    acc = 0
                    for num in cleaned:
                        if int(torch.round(num)) == int(torch.round(batch_y[count])):
                            acc += 10
                        count += 1
                    avg_acc += acc

                avg_acc = avg_acc / iterations
                avg_val_loss = total_val_loss / iterations  # 计算平均验证损失   
                acc_values.append(avg_acc)
                loss_values.append(avg_val_loss)
                writer.add_scalar('Accuracy/val', avg_acc, iters)
                writer.add_scalar('Loss/val', avg_val_loss, iters)  # 记录验证损失 
                print(f'Accuracy/val={avg_acc}         Loss/val={avg_val_loss}')
                # print(f'Accuracy/val=', avg_acc)
                # print('Loss/val=', avg_val_loss)  # 打印验证损失 

                # 更新最高准确率及其迭代次数
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_iter = iters
                    # 保存当前最高准确率到指定文件
                    valid_path = os.path.join(result_path, 'valid')
                    os.makedirs(valid_path, exist_ok=True)
                    # with open(os.path.join(results_path, 'best_accuracy.txt'), 'w') as f:
                    with open(os.path.join(valid_path, 'best_valid_accuracy.txt'), 'a') as f:
                        f.write(f"Iteration: {best_iter}   Best Accuracy: {best_acc}\n")  # 在同一行输出
                        
                # train_set
                iterations = 100
                avg_acc_train = 0

                total_train_loss = 0 

                for _ in range(iterations):
                    batch_x, batch_y = get_batch(batch_size,train_data,val_data,test_data,window_size=10000, split='train')
                    cleaned = model(batch_x)
                    # 计算训练损失
                    train_loss = criterion(cleaned, batch_y)
                    total_train_loss += train_loss.item()  # 累加训练损失

                    count = 0
                    acc = 0
                    for num in cleaned:
                        if int(torch.round(num)) == int(torch.round(batch_y[count])):
                            acc += 10
                        count += 1
                    avg_acc_train += acc
                

                avg_acc_train = avg_acc_train / iterations
                avg_train_loss = total_train_loss / iterations  # 计算平均训练损失
                acc_values_train.append(avg_acc_train)
                loss_values_train.append(avg_train_loss)
                writer.add_scalar('Accuracy/train', avg_acc_train, iters)
                writer.add_scalar('Loss/train', avg_train_loss, iters)  # 记录训练损失
                with open(results_file_path, 'a') as f:
                    f.write(f"Iteration: {iters}, ")
                    f.write(f"Train Accuracy: {avg_acc_train}, Train Loss: {avg_train_loss}, ")
                    f.write(f"Validation Accuracy: {avg_acc}, Validation Loss: {avg_val_loss}\n")

                print(f'Accuracy/train={avg_acc_train}         Loss/train={avg_train_loss}')


        # 保存模型和图表
        if iters % 100 == 0 and iters != 0:
            print("this is the iters:", iters)
            torch.save(model.state_dict(), os.path.join(results_path, 'CNQ_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(results_path, 'CNQ_optim.opt'))

            fig = plt.figure(figsize=(18, 12))
            plt.title(iters)
            plt.plot(acc_values, color="blue")
            plt.plot(acc_values_train, color="red")
            plt.grid()
            fig.savefig(os.path.join(results_path, "CNQ_model_acc.jpeg"))


            fig = plt.figure(figsize=(18, 12))
            plt.title(iters)
            plt.plot(loss_values, color="blue")
            plt.plot(loss_values_train, color="red")
            plt.grid()
            fig.savefig(os.path.join(results_path, "CNQ_model_loss.jpeg"))

            
            checkpoint_path = os.path.join(result_path, 'checkpoint')
            os.makedirs(checkpoint_path, exist_ok=True)
            model_path = os.path.join(checkpoint_path, f'model_{iters}.pth')
            torch.save(model.state_dict(), model_path)

            df = pd.DataFrame({
                'acc_values': acc_values,
                'acc_values_train': acc_values_train,
                'loss_values': loss_values,
                'loss_values_train': loss_values_train
            })

            # 设置文件保存路径
            save_path = os.path.join(results_path, "training_data.csv")

            # 保存为 CSV
            df.to_csv(save_path, index=False)
    plt.close()


    model_dir = os.path.join(result_path, 'checkpoint')

    # 获取所有以 '.pth' 结尾的文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    # 存储每个模型的准确率
    test_acc = 0

    # 遍历每个模型文件
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        # 加载模型
        # model.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(model_path, weights_only=True))
        with torch.no_grad():
            # 验证
            iterations = 500
            avg_acc = 0
            all_preds = []
            all_labels = []
            for _ in range(iterations):
                batch_x, batch_y = get_batch(batch_size,train_data,val_data,test_data,window_size=10000, split='test')
                cleaned = model(batch_x)
                # 将预测和真实标签添加到列表中
                all_labels.extend(torch.round(batch_y).cpu().numpy())  # 将真实标签四舍五入为离散值
                all_preds.extend(torch.round(cleaned).cpu().numpy())   # 将预测值四舍五入为离散值
                count = 0
                acc = 0
                for num in cleaned:
                    if int(torch.round(num)) == int(torch.round(batch_y[count])):
                        acc += 10
                    count += 1
                avg_acc += acc
            avg_acc = avg_acc / iterations
            print(f'模型 {model_file} 的测试集准确率：{avg_acc}')
            if avg_acc > test_acc:
                test_acc = avg_acc
                # 将列表转换为整数类型
                all_labels = np.array(all_labels, dtype=int)
                all_preds = np.array(all_preds, dtype=int)
                cm = confusion_matrix(all_labels, all_preds)
                # 绘制混淆矩阵
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Class 0', 'Class 1'],
                            yticklabels=['Class 0', 'Class 1'])
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.title(f'Confusion Matrix for {model_file}')
                # 保存混淆矩阵图像
                test_path = os.path.join(result_path, 'test')
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                plt.savefig(os.path.join(test_path, f'confusion_matrix_{model_file}.png'))
                plt.close()  # 关闭当前图形
                # 计算指标
                TN = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]
                TP = cm[1][1]
                Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity) if (Precision + Sensitivity) > 0 else 0
                # 保存最佳模型的结果
                with open(os.path.join(test_path, 'best_test_accuracy2.txt'), 'a') as f:
                    f.write(f"model: {model_file}   Best Accuracy: {test_acc}, ")
                    f.write(f"Sensitivity: {Sensitivity}, Specificity: {Specificity}, ")
                    f.write(f"Precision: {Precision}, F1 Score: {F1}\n")  # 在同一行输出


