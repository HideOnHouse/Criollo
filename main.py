import os
import sys

from matplotlib import pyplot as plt
from matplotlib import rc, font_manager

from criollo import Criollo

FONT_PATH = r'C:/Windows/Fonts/malgun.ttf'
FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
rc('font', family=FONT)


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
    plt.title("누가 제일 많이 떠들었을까?")
    plt.bar(*zip(*temp))
    for k, v in temp:
        plt.text(k, v, v, ha='center')
    plt.savefig(f'{result_dir}/user.png')
    plt.tight_layout()
    plt.close()

    temp = sorted(c.count_time().items(), key=lambda x: x[0])
    plt.figure(dpi=200)
    plt.title("어떤 시간에 제일 많이 떠들까?")
    plt.bar(*zip(*temp))
    for k, v in temp:
        plt.text(k, v, v, ha='center')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/time.png')
    plt.close()

    temp = sorted(c.count_text(k=30).items(), key=lambda x: x[1])
    plt.figure(dpi=200)
    plt.title(f"어떤 단어가 제일 많이 나왔을까?")
    plt.barh(*zip(*temp))
    for k, v in temp:
        plt.text(v, k, v, va='center')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/text.png')
    plt.close()

    temp = c.count_text_per_user()
    for user in temp:
        current_user = sorted(temp[user].items(), key=lambda x: x[1])

        # has coda
        if ord(user[-1]) - 44032 % 28 == 0:
            postfix = '가'
        else:
            postfix = '이'

        plt.title(f"{user}{postfix} 제일 많이 사용한 단어는?")
        plt.barh(*zip(*current_user))
        for k, v in current_user:
            plt.text(v, k, v, va='center')
        plt.tight_layout()
        plt.savefig(f"{result_dir}/{user}.png")
        plt.close()


if __name__ == '__main__':
    main(sys.argv)
