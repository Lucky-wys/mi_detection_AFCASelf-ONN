from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

plt.switch_backend("agg")
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
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score

from Data.data_utils_1226 import gen_data, get_batch, get_batch_0409, gen_data2


from Models2.Self_onn_fcanet_ada1 import Self_onn_fcanet_ada1
from Models2.CNN import CNN
from Models2.Self_onn_SE import Self_onn_SE
from Models2.Self_onn import Self_onn
from Models2.Self_onn_fcanet_ada2 import Self_onn_fcanet_ada2
from Models2.CNN_fcanet_ada2 import CNN_fcanet_ada2


def create_incremental_dir_simple(base_path):

    # 确保基础目录存在
    os.makedirs(base_path, exist_ok=True)

    # 查找下一个可用的数字
    num = 1
    while os.path.exists(os.path.join(base_path, str(num))):
        num += 1

    # 创建目录
    target_dir = os.path.join(base_path, str(num))
    os.makedirs(target_dir, exist_ok=True)

    return target_dir


def get_current_datetime():
    # 获取当前的日期和时间
    current_time = datetime.datetime.now()

    # 将日期和时间拼接成字符串
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    # 返回拼接后的时间字符串
    return formatted_time


for i in range(5):
    batch_size = 10
    num_iters = 20000
    window_size = 10000
    exp_name = "Self_onn_fcanet_ada2"
    seed_num = random.randint(0, 100)
    # seed_num = seed_list[i]
    riqi = get_current_datetime()
    base_path = f"Selfonn_ada_results/Self_onn_fcanet_ada2/{seed_num}"
    data_path = create_incremental_dir_simple(base_path)
    result_path = os.path.join(data_path, f"{exp_name}", f"{riqi}")

    train_data, val_data, test_data = gen_data(data_path, seed_num, chns=["ii", "v5"])

    results_path = os.path.join(result_path, "results")

    model = Self_onn_fcanet_ada2().to("cuda:0")
    # model = CNN().to('cuda:0')
    # model.cuda()

    # model = nn.DataParallel(model, device_ids=[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir=results_path)

    batch_size = 10
    acc_values = []
    acc_values_train = []

    loss_values = []
    loss_values_train = []
    # 初始化最高准确率为 0
    best_acc = 0
    best_iter = 0

    results_file_path_plot = os.path.join(result_path, "plot")
    os.makedirs(results_file_path_plot, exist_ok=True)
    results_file_path = os.path.join(results_file_path_plot, "training_results.txt")

    best_loss = float("inf")  # 设置初始最低损失为无穷大
    saved_models = []  # 用于存储保存的模型路径
    saved_models_test = []  # 用于存储测试集上最佳模型路径
    max_saved_models = 10  # 最多保存的模型数量
    best_test_acc = 0

    # 修改后的训练过程
    for iters in range(num_iters):
        batch_x, batch_y = get_batch(
            batch_size, train_data, val_data, test_data, window_size=window_size, split="train"
        )
        batch_x, batch_y = batch_x.to("cuda:0"), batch_y.to("cuda:0")

        y_pred = model(batch_x)

        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        if iters % 100 == 0 and iters != 0:

            writer.add_scalar("Loss/train", loss, iters)
            print("Loss/train=", loss.cpu().detach().numpy())

            with torch.no_grad():

                # validation
                iterations = 100
                avg_acc = 0
                total_val_loss = 0  # 用于累计验证损失   +++

                for _ in range(iterations):
                    batch_x, batch_y = get_batch(
                        batch_size, train_data, val_data, test_data, window_size=window_size, split="val"
                    )

                    cleaned = model(batch_x)

                    # 计算验证损失
                    val_loss = criterion(cleaned, batch_y)  # ++
                    total_val_loss += val_loss.item()  # ++

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
                writer.add_scalar("Accuracy/val", avg_acc, iters)
                writer.add_scalar("Loss/val", avg_val_loss, iters)  # 记录验证损失
                print(f"Accuracy/val={avg_acc}         Loss/val={avg_val_loss}")
                # print(f'Accuracy/val=', avg_acc)
                # print('Loss/val=', avg_val_loss)  # 打印验证损失

                # 更新最高准确率及其迭代次数
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_iter = iters
                    # 保存当前最高准确率到指定文件
                    valid_path = os.path.join(result_path, "valid")
                    os.makedirs(valid_path, exist_ok=True)
                    # with open(os.path.join(results_path, 'best_accuracy.txt'), 'w') as f:
                    with open(os.path.join(valid_path, "best_valid_accuracy.txt"), "a") as f:
                        f.write(f"Iteration: {best_iter}   Best Accuracy: {best_acc}\n")  # 在同一行输出

                # train_set
                iterations = 100
                avg_acc_train = 0

                total_train_loss = 0

                for _ in range(iterations):
                    batch_x, batch_y = get_batch(
                        batch_size, train_data, val_data, test_data, window_size=window_size, split="train"
                    )
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
                writer.add_scalar("Accuracy/train", avg_acc_train, iters)
                writer.add_scalar("Loss/train", avg_train_loss, iters)  # 记录训练损失
                with open(results_file_path, "a") as f:
                    f.write(f"Iteration: {iters}, ")
                    f.write(f"Train Accuracy: {avg_acc_train}, Train Loss: {avg_train_loss}, ")
                    f.write(f"Validation Accuracy: {avg_acc}, Validation Loss: {avg_val_loss}\n")

                print(f"Accuracy/train={avg_acc_train}         Loss/train={avg_train_loss}")
                # scheduler.step(avg_val_loss)

        # 保存模型和图表
        if iters % 100 == 0 and iters != 0:
            print("this is the iters:", iters)
            torch.save(model.state_dict(), os.path.join(results_path, "CNQ_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(results_path, "CNQ_optim.opt"))

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

            checkpoint_path = os.path.join(result_path, "checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            model_path = os.path.join(checkpoint_path, f"model_{iters}.pth")
            torch.save(model.state_dict(), model_path)

            df = pd.DataFrame(
                {
                    "acc_values": acc_values,
                    "acc_values_train": acc_values_train,
                    "loss_values": loss_values,
                    "loss_values_train": loss_values_train,
                }
            )

            # 设置文件保存路径
            save_path = os.path.join(results_path, "training_data.csv")

            # 保存为 CSV
            df.to_csv(save_path, index=False)
    plt.close()

    for j in [500, 300, 100]:
        for i in range(1):
            model_dir = os.path.join(result_path, "checkpoint")

            # 获取所有以 '.pth' 结尾的文件
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

            # 存储每个模型的准确率
            test_acc = 0
            test_path = os.path.join(result_path, f"test{j}_{i}")
            # 创建保存ROC曲线的文件夹
            roc_path = os.path.join(test_path, "roc_curves")
            if not os.path.exists(roc_path):
                os.makedirs(roc_path)

            # 遍历每个模型文件
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                # 加载模型
                model.load_state_dict(torch.load(model_path, weights_only=True))
                with torch.no_grad():
                    # 验证
                    iterations = j
                    avg_acc = 0
                    all_preds = []
                    all_labels = []
                    all_scores = []  # 存储模型输出的得分

                    for _ in range(iterations):
                        batch_x, batch_y = get_batch(
                            batch_size, train_data, val_data, test_data, window_size=window_size, split="test"
                        )
                        cleaned = model(batch_x)
                        all_labels.extend(torch.round(batch_y).cpu().numpy())
                        all_preds.extend(torch.round(cleaned).cpu().numpy())
                        all_scores.extend(cleaned.cpu().numpy())  # 保存模型的得分

                        count = 0
                        acc = 0
                        for num in cleaned:
                            if int(torch.round(num)) == int(torch.round(batch_y[count])):
                                acc += 10
                            count += 1
                        avg_acc += acc

                    avg_acc = avg_acc / iterations
                    print(f"模型 {model_file} 的测试集准确率：{avg_acc}")

                    if avg_acc > test_acc:
                        test_acc = avg_acc
                        all_labels = np.array(all_labels, dtype=int)
                        all_preds = np.array(all_preds, dtype=int)
                        all_scores = np.array(all_scores).flatten()  # 转换为一维数组

                        cm = confusion_matrix(all_labels, all_preds)
                        # 绘制混淆矩阵
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=["Class 0", "Class 1"],
                            yticklabels=["Class 0", "Class 1"],
                        )
                        plt.ylabel("True label")
                        plt.xlabel("Predicted label")
                        plt.title(f"Confusion Matrix for {model_file}")

                        if not os.path.exists(test_path):
                            os.makedirs(test_path)
                        plt.savefig(os.path.join(test_path, f"confusion_matrix_{model_file}.png"))
                        plt.close()

                        # 计算ROC曲线
                        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
                        roc_auc = auc(fpr, tpr)  # 计算AUC

                        roc_data = pd.DataFrame(
                            {"False Positive Rate": fpr, "True Positive Rate": tpr, "Thresholds": thresholds}
                        )
                        roc_data.to_csv(os.path.join(roc_path, f"roc_data_{model_file}.csv"), index=False)
                        # fpr, tpr, _ = roc_curve(all_labels, all_scores)  # 计算假阳性率和真阳性率
                        # roc_auc = auc(fpr, tpr)  # 计算AUC
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
                        plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(f"Receiver Operating Characteristic for {model_file}")
                        plt.legend(loc="lower right")
                        plt.savefig(os.path.join(roc_path, f"roc_curve_{model_file}.png"))
                        plt.close()

                        # 计算指标
                        TN = cm[0][0]
                        FP = cm[0][1]
                        FN = cm[1][0]
                        TP = cm[1][1]
                        Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                        F1 = (
                            2 * (Precision * Sensitivity) / (Precision + Sensitivity)
                            if (Precision + Sensitivity) > 0
                            else 0
                        )
                        Kappa = cohen_kappa_score(all_labels, all_preds)

                        # 保存最佳模型的结果
                        with open(os.path.join(test_path, "best_test_accuracy2.txt"), "a") as f:
                            f.write(f"model: {model_file}   Best Accuracy: {test_acc}, ")
                            f.write(f"Sensitivity: {Sensitivity}, Specificity: {Specificity}, ")
                            f.write(f"Precision: {Precision}, F1 Score: {F1}, Kappa: {Kappa}\n")
