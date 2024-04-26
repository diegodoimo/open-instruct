import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.gridspec import GridSpec

path = "/home/diego/Documents/area_science/ricerca/finetuning_llm/open-instruct/results"
with open(f"{path}/train_statistics_epoch4.pkl", "rb") as f:
    stats = pickle.load(f)


stats.keys()

ov_0shot = [
    stats["ov_0shot"][iter_]["base_model.model.lm_head"]
    for iter_ in stats["iter"].values()
]

ov_5shot = [
    stats["ov_5shot"][iter_]["base_model.model.lm_head"]
    for iter_ in stats["iter"].values()
]

x = [iter_ for iter_ in stats["iter"].values()]

val_accuracy = [stats["mmlu_val"][iter_] for iter_ in stats["iter"].values()]

val_accuracy[0] = 0.40

sns.set_style("whitegrid")
fig = plt.figure(figsize=(6, 3))
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
sns.lineplot(ax=ax, x=x, y=ov_0shot, label="overlap 0 shot", marker="o")
sns.lineplot(ax=ax, x=x, y=ov_5shot, label="overlap 5 shot", marker="o")
ax.set_xlabel("train iteration")
ax.set_title("llama-2-7b")
ax.axvline(x=18, color="black", linestyle="--", linewidth=0.8)
ax.axvline(x=36, color="black", linestyle="--", linewidth=0.8)
ax.axvline(x=54, color="black", linestyle="--", linewidth=0.8)
ax.axvline(x=72, color="black", linestyle="--", linewidth=0.8, label="epochs")
ax.legend(fontsize=9)
ax.set_ylabel("overlap", fontsize=12)

ax = fig.add_subplot(gs[1])
sns.lineplot(ax=ax, x=x, y=val_accuracy, label="finetune accuracy", marker="o")
ax.set_ylabel("mmlu (val) accuracy")
ax.set_xlabel("train iteration")
ax.set_title("llama-2-7b")
gs.tight_layout(fig)
plt.savefig("./overlap_finetuned_4epochs.png", dpi=150)
