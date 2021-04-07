import numpy as np
import pygame
import pygame.locals
import sys
import time

'''
observation = env.reset()
重置棋局

env.action_space			
返回棋局双方每个棋子对应action_space

sample_action = env.action_sapce.sample()    	
首先获取当前的棋手的标记，选择当前棋手的存活棋子的当前可行的action_space的sample

env.observation_space			
返回棋局的表示空间（棋局有两种表示：1.用正负表示双方，每种棋子用一个数字表示，整个棋面是10*9
							  2.用正负1表示双方，7个平面重叠分别依次表示七种棋子
				)

observation , reward , done , info = env.step()
输入action(仅仅表示当前应行子棋手的动作)
observation为当前棋局，（棋局有两种表示：1.用正负表示双方，每种棋子用一个数字表示，整个棋面是9*10
                        2.用正负1表示双方，7个平面重叠分别依次表示七种棋子
                      )
reward:
done:
info:0,1代表此步是否有效，2代表棋子位置为空，3代表棋子棋手不匹配

env.render()
将当前棋面展示出来

env.close()				
关闭展示出来的棋面

action	 				
输入是要行的棋子，以及此子的棋盘坐标，若是输入动作无效，则此步无效
(chess_pieces_position,move_position)
'''


class Action_space():
    def __init__(self):
        pass

    def sample(self):
        pass

    def __repr__(self):
        pass


class Observation_space():
    def __init__(self):
        pass

    def __repr__(self):
        pass


class CC():
    ini_borad = np.array([
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 0, 0, 6, 0],
        [7, 0, 7, 0, 7, 0, 7, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-7, 0, -7, 0, -7, 0, -7, 0, -7],
        [0, -6, 0, 0, 0, 0, 0, -6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -2, -3, -4, -5, -4, -3, -2, -1],
    ])
    pieces_dict = {1: '车', 2: '马', 3: '相', 4: '仕', 5: '将', 6: '炮', 7: '士',
                   -1: '车', -2: '马', -3: '相', -4: '仕', -5: '将', -6: '炮', -7: '士', }

    def __init__(self, board_code_type=1):
        self.chess_player = 1  # 1红(0-44)  -1黑(45-89)
        self.board = np.zeros((10, 9))  # 10*9棋盘
        self.board_code_type = board_code_type  # 1 represent 10*9  2 represent7*10*9
        self.action_space = Action_space()
        self.observation_space = Observation_space()
        self.move_error_reward = -1
        self.reset()

    def board_code(self):  # 对应board的编码，board_code_type=2似乎更利于训练
        board = self.board.copy()
        if self.board_code_type == 1:  # 返回数据是np.int
            return board.astype(np.int)
        elif self.board_code_type == 2:
            a = (board == 1).astype(np.int)
            b = (board == -1).astype(np.int)
            b[b == 1] = -1
            stack_board = a + b
            for i in range(2, 8):
                a = (board == i).astype(np.int)[np.newaxis]
                b = (board == -i).astype(np.int)[np.newaxis]
                b[b == -i] = -1
                stack_board = np.vstack((stack_board, a + b))

            return stack_board

    def reset(self):
        self.board = CC.ini_borad.copy()
        self.chess_player = 1
        observation = self.board_code()
        return observation

    def step(self, action):
        """
        1.判断行子者是否是当前棋手
            其次判断棋子位置和移动位置的棋子是否是同一方，判断棋子将行的目标位置是否可行
                若是可行，移动棋子到目标位置
	                行子后判断对方将是否还在，若在则done为False,不在done为True
	                改应行棋手为另一方
	                返回信息，结束当前step
                若是不可行，返回信息结束当前step，棋盘状态未改变，应行子棋手未改变
        2.判断棋子位置是否有棋子
        3.所行棋子不是当前棋手的棋子

        :param action: action为tuple,action[0]为所行棋子位置，action[1]为将行位置
        :return: observation, reward, done, info
        info: 0表示将行棋子与将行位置上的棋子的乘积，>0表示两个位置的棋子为同一方
                或表示chess_pieces_position和move_position为同一位置
              1表示行棋成功
              2表示棋子位置为空
              3表示棋手和棋子不匹配
        """
        done = False
        reward = 0
        info = None
        observation = None
        chess_pieces_position = action[0]
        move_position = action[1]
        chess_pieces = self.board[chess_pieces_position]
        move_pieces = self.board[move_position]
        if chess_pieces * self.chess_player > 0:  # 将行棋子是否是当前应执棋者
            if chess_pieces * move_pieces > 0:  # 将行棋子与将行位置上的棋子的乘积，>0表示两个位置的棋子为同一方
                info = 0  # 或表示chess_pieces_position和move_position为同一位置
            else:
                info = int(self.get_do_position(chess_pieces_position, move_position))

            if info:
                if abs(self.board[move_position]) == 5:
                    done = True
                if self.board[move_position] * self.chess_player == 1:
                    reward = 1
                self.board[move_position] = chess_pieces
                self.board[chess_pieces_position] = 0
            else:
                reward = self.move_error_reward
            observation = self.board_code().copy()
        elif chess_pieces * self.chess_player == 0:
            observation = self.board_code().copy()
            reward = self.move_error_reward
            done = False
            info = 2  # 代表棋子位置为空
        else:
            observation = self.board_code().copy()
            reward = self.move_error_reward
            done = False
            info = 3  # 代表棋手和棋子不匹配
        if info == 1:
            self.chess_player = -1 * self.chess_player
            print(move_position)
        print(done)
        return observation, reward, done, info

    def get_do_position(self, chess_pieces_postion, move_position):
        """
        传入位置已经经过
        1.是否将行棋子为空
        2.将行棋子是否是当前应执棋者
        3.将行棋子的位置和将行位置上的棋子是否属于同一方
        判断，然后

        首先根据chess_pieces_position确定棋子的种类
            根据棋子的种类和位置确定其在该位置下可行的
            黑红双方在各个种类中自己判断，因为有些棋子有通用规则

        :param chess_pieces_postion: 棋子的位置
        :param move_position: 棋子将行的位置
        :return: 棋子到将行的位置是否可行 int(info)
        """

        assert type(chess_pieces_postion) == tuple and type(move_position) == tuple
        info = None
        chess_pieces = CC.pieces_dict[self.board[chess_pieces_postion]]
        # do_position = None
        if chess_pieces == '车':  # 车
            '''
            因为车的可行位置一定在某一轴与车目前位置相同，所以
            首先获取move_position的横纵坐标，
            若move_position的横纵坐标有一维与chess_pieces_position相同
                查看move_position不相同的一维与chess_pieces_position在此维度
                之间的位置是否有棋子
                    若有info = False
            否则表示move_position不在车的可行范围内:
                info = False
            '''
            move_row = move_position[0]
            move_column = move_position[1]
            if move_row == chess_pieces_postion[0]:
                a = move_column - chess_pieces_postion[1]
                for i in range(int(a / abs(a)), int(a), int(a / abs(a))):  # 去头去尾
                    position = (move_row, i + chess_pieces_postion[1])
                    if self.board[position] != 0:
                        info = False
                        break
                else:
                    info = True
            elif move_column == chess_pieces_postion[1]:
                a = move_row - chess_pieces_postion[0]
                for i in range(int(a / abs(a)), int(a), int(a / abs(a))):  # 去头去尾
                    position = (i + chess_pieces_postion[0], move_column)
                    if self.board[position] != 0:
                        info = False
                        break
                else:
                    info = True
            else:
                info = False

        elif chess_pieces == '马':  # 马
            '''
            因为马的每个坐标位置理论上都可以行进四个位置即上下左右(有些位置超出棋盘)
            通过坐标来做，先确定马当前位置可以行进的位置，再判断蹩脚马情况，
            最后看move_position在不在do_position
            棋子位置向上:row-2,column+-1
                   向下:row+2,column+-1
                   向左:row+-1,column-2
                   向右:row+-1,column+2
            '''
            row = chess_pieces_postion[0]
            column = chess_pieces_postion[1]
            do_position = [(row - 2, column + 1), (row - 2, column - 1), (row + 2, column + 1), (row + 2, column - 1),
                           (row + 1, column - 2), (row - 1, column - 2), (row + 1, column + 2), (row - 1, column + 2)]
            try:
                index = do_position.index(move_position) // 2  # move_position可能不在里面，防止出错
                if index == 0:
                    if self.board[row - 1, column] != 0:
                        info = False
                    else:
                        info = True
                elif index == 1:
                    if self.board[row + 1, column] != 0:
                        info = False
                    else:
                        info = True
                elif index == 2:
                    if self.board[row, column - 1] != 0:
                        info = False
                    else:
                        info = True
                else:
                    if self.board[row, column + 1] != 0:
                        info = False
                    else:
                        info = True
            except ValueError:
                info = False

        elif chess_pieces == '相':  # 相
            '''
            由于相的位置的比较少，就可以直接用dict来表示相的每个位置的可行位置
            首先获取将行棋子的位置，然后在dict中寻找到相应的可行位置，
            之后再判断这些位置是否有效：
            1.先判断move_position是否在dict的key列表中
                若不在 info=False
                若在 获取chess_pieces_position和move_position的田字格中间位置，看是否有棋子
            '''
            xiang_possible_position = {(0, 2): [(2, 0), (2, 4), ], (0, 6): [(2, 4), (2, 8)],
                                       (2, 0): [(0, 2), (4, 2)], (2, 4): [(0, 2), (0, 6), (4, 2), (4, 6)],
                                       (2, 8): [(0, 6), (4, 6)], (4, 2): [(2, 0), (2, 4)], (4, 6): [(2, 4), (2, 8)],
                                       (9, 2): [(7, 0), (7, 4)], (9, 6): [(7, 4), (7, 8)],
                                       (7, 0): [(9, 2), (5, 2)], (7, 4): [(9, 2), (9, 6), (5, 2), (5, 6)],
                                       (7, 8): [(9, 6), (5, 6)], (5, 2): [(7, 0), (7, 4)], (5, 6): [(7, 4), (7, 8)], }
            try:
                do_position = xiang_possible_position[chess_pieces_postion]
            except KeyError:
                info = False
            else:
                if move_position not in do_position:
                    info = False
                else:
                    mid_row = int((chess_pieces_postion[0] + move_position[0]) / 2)
                    mid_column = int((chess_pieces_postion[1] + move_position[1]) / 2)
                    middle_position = (mid_row, mid_column)
                    if self.board[middle_position] == 0:
                        info = True
                    else:
                        info = False

        elif chess_pieces == '仕':  # 仕
            '''
            由于仕的位置相对较少，所以采用dict直接构建可行位置即可。
            由于传入的两个位置已经保证了不会属于同一方，所以若将行位置在shi_possible_position的key中即可
            '''
            shi_possible_position = {
                (0, 3): [(1, 4), ], (0, 5): [(1, 4), ],
                (1, 4): [(0, 3), (0, 5), (2, 3), (2, 5)],
                (2, 3): [(1, 4), ], (2, 5): [(1, 4), ],
                (9, 3): [(8, 4), ], (9, 5): [(8, 4), ],
                (8, 4): [(9, 3), (9, 5), (7, 3), (7, 5)],
                (7, 3): [(8, 4), ], (7, 5): [(8, 4), ],
            }
            try:
                do_position = shi_possible_position[chess_pieces_postion]
            except KeyError:
                info = False
            else:
                if move_position not in do_position:
                    info = False
                else:
                    info = True

        elif chess_pieces == '将':  # 将
            '''
            首先判断move_position是否在双方将棋可行的位置区域
            若是可以，则利用将每次只能上下左右行进一步的规矩，计算
            move_position和chess_pieces_position的横纵坐标的
            差的绝对值的和，只有等于才可行
            另一种情况是两将相对，其间没有棋子相隔，可以直接吃将

            '''
            move_row = move_position[0]
            move_column = move_position[1]
            if (2 < move_row < 7) or (move_column < 3 or move_column > 5):
                info = False
            else:
                judge = abs(chess_pieces_postion[0] - move_row) + abs(chess_pieces_postion[1] - move_column)
                if judge == 1:
                    info = True
                else:
                    info = False

            if abs(self.board[move_position]) == 5 and chess_pieces_postion[1] == move_column:
                a = move_row - chess_pieces_postion[0]
                for i in range(int(a / abs(a)), int(a), int(a / abs(a))):  # 去头去尾
                    position = (i + chess_pieces_postion[0], move_column)
                    if self.board[position] != 0:
                        break
                else:
                    info = True

        elif chess_pieces == '炮':  # 炮
            '''
            因为炮的可行位置必定和炮在某一维相同,所以
            获取move_position的横纵坐标，
            若move_position的横纵坐标有一维与chess_pieces_position相同
                遍历两者之间的位置，查看其间棋子个数
                判断move_position的位置是否是敌方棋子，
                若是
                    看两者之间是否只有一个棋子，
                        若是 info = True
                        若不是 info = False  因为炮不能不隔山就打炮
                若不是
                    看两者之间是否有棋子，
                        若是 info = False
                        若不是 info = True
            '''
            move_row = move_position[0]
            move_column = move_position[1]
            judge = self.board[move_position] * self.board[chess_pieces_postion] < 0  # move_postion是否是敌方
            if move_row == chess_pieces_postion[0]:
                a = move_column - chess_pieces_postion[1]
                pieces_num = 0
                for i in range(int(a / abs(a)), int(a), int(a / abs(a))):
                    position = (move_row, i + chess_pieces_postion[1])
                    if self.board[position] != 0:
                        pieces_num = pieces_num + 1

                if judge and pieces_num == 1:
                    info = True
                else:
                    if not judge and pieces_num == 0:
                        info = True
                    else:
                        info = False
            elif move_column == chess_pieces_postion[1]:
                a = move_row - chess_pieces_postion[0]
                pieces_num = 0
                for i in range(int(a / abs(a)), int(a), int(a / abs(a))):
                    position = (i + chess_pieces_postion[0], move_column)
                    if self.board[position] != 0:
                        pieces_num = pieces_num + 1
                if judge and pieces_num == 1:
                    info = True
                else:
                    if not judge and pieces_num == 0:
                        info = True
                    else:
                        info = False
            else:
                info = False

        elif chess_pieces == '士':  # 士
            '''
            分为红黑双方棋盘，红方row(0-4),黑方row(5-9)
            先获取chess_pieces_position和move_position之间横纵坐标的差的绝对值之和
                若>1
                    info = False
                否则
                    通过boundary_dict 获取相应行棋方的楚河汉界线，
                    若self.chess_player * chess_pieces_position row <= 
                    self.chess_player * 楚河汉界线(由于黑红方判断士是否在自己地盘内的方式不一样，红<=,黑>=
                    所以乘以self.chess_player黑负红正,红加黑减同理可用)
                        判断move_position是否是chess_pieces_position的row前进一步(红加黑减)
                            若是 info = True
                            若不是 info = False
                    若self.chess_player * chess_pieces_position row > 
                    self.chess_player * 楚河汉界线
                        判断move_position是否是chess_pieces_position的row后退一步(红减黑加)
                            若是 info = False
                            若不是 info = True
            '''
            boundary_dict = {1: 4, -1: 5}
            judge = abs(chess_pieces_postion[0] - move_position[0]) + \
                    abs(chess_pieces_postion[1] - move_position[1])
            if judge > 1:
                info = False
            else:
                boundary = boundary_dict[self.chess_player]
                if self.chess_player * chess_pieces_postion[0] <= self.chess_player * boundary:
                    do_position = (chess_pieces_postion[0] + self.chess_player, chess_pieces_postion[1])
                    if do_position == move_position:
                        info = True
                    else:
                        info = False
                else:
                    do_position = (chess_pieces_postion[0] - self.chess_player, chess_pieces_postion[1])
                    if do_position == move_position:
                        info = False
                    else:
                        info = True

        else:
            raise ValueError

        return info

    def render(self):
        print(self.board)

    def close(self):
        pass


# env = CC()
# env.reset()
# env.step(((2,1),(7,1)))

class CC_board:
    """
    总共棋盘的长宽尺寸比为10：12，所以棋盘大小为10*UNIT_PIXEL * 13*UNIT_PIXEL
    宽(x轴)为8个棋盘格子加上棋盘两边各1个格子，棋盘竖线应该为9条，第一条从X轴1*UNIT_PIXEL开始，
    结束为9*UNIT_PIXEL，其中除去第一条和第九条竖线长为9*UNIT_PIXEL，其余各条由于楚河汉界的存在，
    需要将在楚河汉界的线段涂成背景色
    长(y轴)为9个棋盘格子加上上下的边界各一个格子，以及最上面的2个格子显示对战信息.
    棋盘横线应该为10条，第一条应该从Y轴2*UNIT_PIXEL开始，结束为11*UNIT_PIXEL，长为8*UNIT_PIXEL


    """
    UNIT_PIXEL = 40
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

    def __init__(self, ):
        pygame.init()
        self.CC = CC()

        self.board = pygame.display.set_mode((10 * CC_board.UNIT_PIXEL, 12 * CC_board.UNIT_PIXEL))
        pygame.display.set_caption('中国象棋')
        self.board.fill(CC_board.WHITE)
        # 绘制棋盘
        pygame.draw.line(self.board,
                         CC_board.BLACK,
                         (1 * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                         (1 * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL), )
        pygame.draw.line(self.board,
                         CC_board.BLACK,
                         (9 * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                         (9 * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL), )
        for i in range(2, 9):
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (i * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                             (i * CC_board.UNIT_PIXEL, 6 * CC_board.UNIT_PIXEL))
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (i * CC_board.UNIT_PIXEL, 7 * CC_board.UNIT_PIXEL),
                             (i * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL))
        for i in range(2, 12):
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (1 * CC_board.UNIT_PIXEL, i * CC_board.UNIT_PIXEL),
                             (9 * CC_board.UNIT_PIXEL, i * CC_board.UNIT_PIXEL))
        self.draw()
        pygame.display.update()

    def draw(self):
        """
        依据CC.board来绘制棋盘，每次行棋之后都需要进行棋盘的绘制
        :return:None
        """
        self.board = pygame.display.set_mode((10 * CC_board.UNIT_PIXEL, 12 * CC_board.UNIT_PIXEL))
        pygame.display.set_caption('中国象棋')
        self.board.fill(CC_board.WHITE)
        # 绘制棋盘
        pygame.draw.line(self.board,
                         CC_board.BLACK,
                         (1 * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                         (1 * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL), )
        pygame.draw.line(self.board,
                         CC_board.BLACK,
                         (9 * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                         (9 * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL), )
        for i in range(2, 9):
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (i * CC_board.UNIT_PIXEL, 2 * CC_board.UNIT_PIXEL),
                             (i * CC_board.UNIT_PIXEL, 6 * CC_board.UNIT_PIXEL))
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (i * CC_board.UNIT_PIXEL, 7 * CC_board.UNIT_PIXEL),
                             (i * CC_board.UNIT_PIXEL, 11 * CC_board.UNIT_PIXEL))
        for i in range(2, 12):
            pygame.draw.line(self.board,
                             CC_board.BLACK,
                             (1 * CC_board.UNIT_PIXEL, i * CC_board.UNIT_PIXEL),
                             (9 * CC_board.UNIT_PIXEL, i * CC_board.UNIT_PIXEL))

        player_dict = {1: CC_board.RED, -1: CC_board.BLACK, }
        player_do_dict = {1: '红方行子', -1: '黑方行子'}
        radius = int(0.37 * CC_board.UNIT_PIXEL)
        for i in range(10):
            for j in range(9):
                if self.CC.board[i][j] != 0:
                    color = player_dict[self.CC.board[i][j] / abs(self.CC.board[i][j])]
                    piece = self.CC.pieces_dict[self.CC.board[i][j]]
                    pos = ((j + 1) * CC_board.UNIT_PIXEL, (i + 2) * CC_board.UNIT_PIXEL)
                    circle = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(circle, CC_board.YELLOW, pos, radius)
                    fontObj = pygame.font.SysFont('SimHei', 24)
                    textSurfaceObj = fontObj.render(piece, True, color, CC_board.YELLOW)
                    RecObj = textSurfaceObj.get_rect()
                    RecObj.center = pos
                    # circle.blit(textSurfaceObj,RecObj)
                    self.board.blit(circle, pos)
                    self.board.blit(textSurfaceObj, RecObj)
        player_do = player_do_dict[self.CC.chess_player]
        fontObj = pygame.font.SysFont('SimHei', 24)
        textSurfaceObj = fontObj.render(player_do, True, CC_board.BLACK, CC_board.WHITE)
        RecObj = textSurfaceObj.get_rect()
        RecObj.center = (5 * CC_board.UNIT_PIXEL, 1 * CC_board.UNIT_PIXEL)
        self.board.blit(textSurfaceObj, RecObj)
        pygame.display.update()

    def get_position(self, position):
        x = (position[0] - 1 * CC_board.UNIT_PIXEL) // CC_board.UNIT_PIXEL
        y = (position[1] - 2 * CC_board.UNIT_PIXEL) // CC_board.UNIT_PIXEL
        x1 = position[0] % CC_board.UNIT_PIXEL
        y1 = position[1] % CC_board.UNIT_PIXEL
        if x1 >= 0.6 * CC_board.UNIT_PIXEL:
            x = x + 1
        if y1 >= 0.6 * CC_board.UNIT_PIXEL:
            y = y + 1
        return (y, x)

    def run(self):
        action = []
        done = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    pygame.quit()
                if event.type == pygame.locals.MOUSEBUTTONUP and event.button == 1:
                    position = self.get_position(event.pos)
                    print(event.pos, position)
                    action.append(position)
                    if len(action) == 1:
                        fontObj = pygame.font.SysFont('SimHei', 24)
                        textSurfaceObj = fontObj.render('起手', True, CC_board.BLACK, CC_board.WHITE)
                        RecObj = textSurfaceObj.get_rect()
                        RecObj.center = (8 * CC_board.UNIT_PIXEL, 1 * CC_board.UNIT_PIXEL)
                        self.board.blit(textSurfaceObj, RecObj)
                        pygame.display.update()
                    elif len(action) == 2:
                        o, r, done, i = self.CC.step(action)
                        action = []
                        pygame.event.clear()
                        self.draw()
            if done:
                print('棋局结束')
                input('按任意键以重新开始,或按X按钮退出')
                self.CC.reset()
                self.draw()
                done = False


chess = CC_board()
chess.run()
