import os
import sys

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import rc

from criollo import Criollo

rc('font', family='nanumgothic')


def main(args):
    if len(args) != 2:
        print("""Usage:
        python criollo.py {kakaotalk_chat_path}
        """)
        return 0

    c = Criollo(args[1])
    result_dir = f"./{c.room_name}"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    temp = c.count_user().items()
    plt.figure(dpi=200)
    plt.title("가장 많이 이야기한 사람")
    plt.bar(*zip(*temp))
    for k, v in temp:
        plt.text(k, v, v, ha='center')
    plt.savefig(f'{result_dir}/user.png')
    plt.tight_layout()
    plt.close()

    temp = sorted(c.count_time().items(), key=lambda x: x[0])
    plt.figure(dpi=200)
    plt.title("활동 시간")
    plt.bar(*zip(*temp))
    for k, v in temp:
        plt.text(k, v, v, ha='center')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/time.png')
    plt.close()

    temp = c.count_time_per_user()
    for user, hist in temp.items():
        hist = sorted(hist.items(), key=lambda x: x[0])
        plt.figure(dpi=200)
        plt.title(f"{user}님의 활동 시간")
        plt.bar(*zip(*hist))
        for k, v in hist:
            plt.text(k, v, v, ha='center')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/{user}_time.png')
        plt.close()

    temp = sorted(c.count_text(k=30).items(), key=lambda x: x[1])
    plt.figure(dpi=200)
    plt.title(f"가장 많이 나온 단어")
    plt.barh(*zip(*temp))
    for k, v in temp:
        plt.text(v, k, v, va='center')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/text.png')
    plt.close()

    temp = c.count_text_per_user()
    for user in temp:
        current_user = sorted(temp[user].items(), key=lambda x: x[1])
        plt.title(f"{user}님이 제일 많이 사용한 단어")
        plt.barh(*zip(*current_user))
        for k, v in current_user:
            plt.text(v, k, v, va='center')
        plt.tight_layout()
        plt.savefig(f"{result_dir}/{user}.png")
        plt.close()

    temp = c.graph_relation(window_size=7)
    g = nx.Graph()
    for k, v in temp.items():
        g.add_nodes_from(k)
        g.add_edge(k[0], k[1], weight=v)
    plt.figure(figsize=(10, 10))
    label_list = [i for i in g.nodes()]
    weight_list = [i[2]['weight'] for i in g.edges(data=True)]
    max_weight = max(weight_list) / 5
    positions = nx.circular_layout(g)
    node_size = max(len(i) for i in label_list)
    nx.draw(g,
            width=[i / max_weight for i in weight_list],
            with_labels=True,
            pos=positions,
            node_size=node_size * 1000,
            node_color='cornflowerblue',
            font_family='NanumGothic')
    plt.savefig('./graph_relation.png')

    # temp = c.sent_cls()
    # for k, v in temp.items():
    #     plt.figure(figsize=(12, 4))
    #     plt.title(f"{k}의 시간별 기분의 변화")
    #     v = pd.DataFrame(v)
    #     v = v.ewm(alpha=0.05).mean()
    #     plt.plot(v)
    #     plt.savefig(f"{result_dir}/{k}_sent.png")
    #     plt.close()


if __name__ == '__main__':
    main(sys.argv)
