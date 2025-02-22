import numpy as np
from torch.utils.data.dataset import Dataset


class YinYangDataset(Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42):
        super(YinYangDataset, self).__init__()
        # using the numpy RNG to allow compatibility to other deep learning frameworks
        np.random.seed(seed)
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']

        # goal_classes = np.random.randint(3, size=size)
        # for cls in range(2):
        #     cls_idx = np.argwhere(goal_classes == cls)
        #     n_class = cls_idx.size
        #     phi = np.random.uniform(-np.pi, np.pi, n_class)
        #     rho = np.random.uniform(-r_big, r_big, n_class)
        #     x, y = rho * np.cos(phi), rho * np.sin(phi)

        for goal_class in np.random.randint(3, size=size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            x, y, c = self.get_sample(goal_class)
            # add mirrod axis values
            x_flipped = 2*r_big - x
            y_flipped = 2*r_big - y
            val = np.array([x, y, x_flipped, y_flipped, 0.9*r_big])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = np.random.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        return self.__vals[index], self.__cs[index]

    def __len__(self):
        return len(self.__cs)
