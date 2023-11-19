import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

# 主观判断
# calculate_part = [
#     [[110, 200], ],
#     [[0,   265], ],
#     [[180, 375], [0, 300]],
#     [[0,   234], ],
#     [[0,   240], [0, 240]],
#     [[35,  166], ],
#     [[0,   166], ],
#     [[0,   210], [0, 180]],
#     [[0,   270], ],     # 图像最右边可能有问题
#     [[30,  355], [16, 340]]
# ]

# 企业给的狭窄段划分数据(修正)
calculate_part = [
    [[11, 328], ],
    [[0,   396], ],
    [[0, 372], [28, 247]],
    [[67,   234], ],
    [[0,   202], [0, 240]],  # 第二个和沃夫曼的不一样
    [[0,    273], ],
    [[0,   415], ],
    [[39,   199], [66, 138]],
    [[68,   261], ],
    [[25,  396], [47, 396]],  # 第一个和沃夫曼的不一样
]

# 企业给的数据(无修正)
# calculate_part = [
#     [[11, 328], ],
#     [[0,   396], ],
#     [[0, 372], [28, 247]],
#     [[67,   234], ],
#     [[0,   202], [0, 341]],  # 第二个右侧到了过于狭窄的地方
#     [[0,    273], ],
#     [[0,   415], ],
#     [[39,   199], [66, 138]],
#     [[68,   261], ],
#     [[25,  155], [47, 396]],  # 第二个到了一个偏大的地方
# ]

# 当前最佳划分(上面上取最好的)
# calculate_part = [
#     [[110, 200], ],
#     [[0,   265], ],
#     [[0, 372], [28, 247]],
#     [[67,   234], ],
#     [[0,   202], [0, 240]],  # 第二个右侧到了过于狭窄的地方
#     [[0,    273], ],
#     [[0,   166], ],
#     [[39,   199], [66, 138]],
#     [[0,   270], ],
#     [[25,  155], [47, 396]],  # 第二个到了一个偏大的地方
# ]

data_dir = Path("./processed_data_2")
plot_dir = Path("./processed_data_2_plot")
(area_dir := plot_dir / "area").mkdir(parents=True, exist_ok=True)
(eccentricity_dir := plot_dir / "eccentricity").mkdir(parents=True, exist_ok=True)
for person_dir in data_dir.iterdir():
    person_id = int(person_dir.stem)-1
    for csv_path in person_dir.iterdir():
        inspection_id = int(csv_path.stem.split('_')[-1])-1
        df = pd.read_csv(csv_path)
        title = f"{person_dir.stem}_{inspection_id+1}"

        # part = calculate_part[person_id][inspection_id]
        ax = df.plot(y='Area', title=title, figsize=(6, 2))
        # ax.vlines(part, 0, np.max(df[['Area']].to_numpy()), linestyles='dashed', colors='red')
        fig = ax.get_figure()
        plt.show()
        fig.savefig(area_dir / f"{title}.png")

        # ax = df.plot(x='OFR_ID', y='Eccentricity', title=title)
        # fig = ax.get_figure()
        # fig.savefig(eccentricity_dir / f"{title}.png")
        # plt.close('all')
