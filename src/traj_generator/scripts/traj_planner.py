'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-01 23:16:14
'''

import numpy as np


class MinJerkPlanner():
    '''
    Minimum-jerk trajectory generator
    '''

    def plan(self, s, head_state, tail_state, int_wpts, ts):
        '''
        Input:
        s: degree of trajectory, 2(acc)/3(jerk)/4(snap)
        head_state: (s,D) array, where s=2(acc)/3(jerk)/4(snap), D is the dimension
        tail_state: the same as head_state
        int_wpts : (M-1,D) array, where M is the piece num of trajectory

        Note about head_state and tail_state:

        If you want a trajecory of minimum s, only the 0~(s-1) degree bounded contitions and valid.
        e.g. for minimum jerk(s=3) trajectory, we can specify
        pos(s=0), vel(s=1), and acc(s=2) in head_state and tail_state both.
        If you provide less than s degree bounded conditions, the missing ones will be set to 0.
        If you provide more than s degree bounded conditions, the extra ones will be ignored.

        head_state and tail_state are supposed to be 2-dimensional arrays.

        The function stores the coeffs in self.coeffs
        '''
        self.s = s
        self.D = head_state.shape[1]
        self.M = len(ts)

        input_head_shape0 = head_state.shape[0]
        input_tail_shape0 = tail_state.shape[0]

        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))

        for i in range(min(self.s, input_head_shape0)):
            self.head_state[i] = head_state[i]
        for i in range(min(self.s, input_tail_shape0)):
            self.tail_state[i] = tail_state[i]

        self.int_wpts = int_wpts  # 'int' for 'intermediate'
        self.ts = ts
        self.get_coeffs(int_wpts, ts)

    def get_coeffs(self, int_wpts, ts):
        '''
        Calculate coeffs according to int_wpts and ts
        input: int_wpts(D,M-1) and ts(M,)
        stores self.A and self.coeffs
        '''
        T1 = ts
        T2 = ts**2
        T3 = ts**3
        T4 = ts**4
        T5 = ts**5

        A = np.zeros((2 * self.M * self.s, 2 * self.M * self.s))
        b = np.zeros((2 * self.M * self.s, self.D))
        b[0:self.s, :] = self.head_state
        b[-self.s:, :] = self.tail_state

        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 2.0

        for i in range(self.M - 1):
            A[6 * i + 3, 6 * i] = 1.0
            A[6 * i + 3, 6 * i + 1] = T1[i]
            A[6 * i + 3, 6 * i + 2] = T2[i]
            A[6 * i + 3, 6 * i + 3] = T3[i]
            A[6 * i + 3, 6 * i + 4] = T4[i]
            A[6 * i + 3, 6 * i + 5] = T5[i]
            A[6 * i + 4, 6 * i] = 1.0
            A[6 * i + 4, 6 * i + 1] = T1[i]
            A[6 * i + 4, 6 * i + 2] = T2[i]
            A[6 * i + 4, 6 * i + 3] = T3[i]
            A[6 * i + 4, 6 * i + 4] = T4[i]
            A[6 * i + 4, 6 * i + 5] = T5[i]
            A[6 * i + 4, 6 * i + 6] = -1.0
            A[6 * i + 5, 6 * i + 1] = 1.0
            A[6 * i + 5, 6 * i + 2] = 2 * T1[i]
            A[6 * i + 5, 6 * i + 3] = 3 * T2[i]
            A[6 * i + 5, 6 * i + 4] = 4 * T3[i]
            A[6 * i + 5, 6 * i + 5] = 5 * T4[i]
            A[6 * i + 5, 6 * i + 7] = -1.0
            A[6 * i + 6, 6 * i + 2] = 2.0
            A[6 * i + 6, 6 * i + 3] = 6 * T1[i]
            A[6 * i + 6, 6 * i + 4] = 12 * T2[i]
            A[6 * i + 6, 6 * i + 5] = 20 * T3[i]
            A[6 * i + 6, 6 * i + 8] = -2.0
            A[6 * i + 7, 6 * i + 3] = 6.0
            A[6 * i + 7, 6 * i + 4] = 24.0 * T1[i]
            A[6 * i + 7, 6 * i + 5] = 60.0 * T2[i]
            A[6 * i + 7, 6 * i + 9] = -6.0
            A[6 * i + 8, 6 * i + 4] = 24.0
            A[6 * i + 8, 6 * i + 5] = 120.0 * T1[i]
            A[6 * i + 8, 6 * i + 10] = -24.0

            b[6 * i + 3] = int_wpts[i]

        A[-3, -6] = 1.0
        A[-3, -5] = T1[-1]
        A[-3, -4] = T2[-1]
        A[-3, -3] = T3[-1]
        A[-3, -2] = T4[-1]
        A[-3, -1] = T5[-1]
        A[-2, -5] = 1.0
        A[-2, -4] = 2 * T1[-1]
        A[-2, -3] = 3 * T2[-1]
        A[-2, -2] = 4 * T3[-1]
        A[-2, -1] = 5 * T4[-1]
        A[-1, -4] = 2.0
        A[-1, -3] = 6 * T1[-1]
        A[-1, -2] = 12 * T2[-1]
        A[-1, -1] = 20 * T3[-1]

        self.A = A

        self.coeffs = np.linalg.solve(A, b)

    def get_pos(self, t):
        '''
        get position at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_pos(sum(self.ts))

        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([1, T, T**2, T**3, T**4, T**5])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_vel(self, t):
        '''
        get velocity at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_vel(sum(self.ts))

        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_acc(self, t):
        '''
        get acceleration at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_acc(sum(self.ts))

        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 2, 6*T, 12*T**2, 20*T**3])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_jerk(self, t):
        '''
        get jerk at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_jerk(sum(self.ts))

        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 0, 6, 24*T, 60*T**2])

        return np.dot(c_block.T, np.array([beta]).T).T
    
    def get_snap(self, t):
        '''
        get snap at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_snap(sum(self.ts))

        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 0, 0, 24, 120*T])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_pos_array(self):
        '''
        return the full pos array
        '''
        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        pos_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            pos_array[i] = self.get_pos(t_samples[i])

        return pos_array

    def get_vel_array(self):
        '''
        return the full vel array
        '''
        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        vel_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            vel_array[i] = self.get_vel(t_samples[i])

        return vel_array

    def get_acc_array(self):
        '''
        return the full acc array
        '''
        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        acc_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            acc_array[i] = self.get_acc(t_samples[i])

        return acc_array

    def get_jer_array(self):
        '''
        return the full jer array
        '''
        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        jer_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            jer_array[i] = self.get_jerk(t_samples[i])

        return jer_array
    
    def get_snap_array(self):
        '''
        return the full snap array
        '''
        if not hasattr(self, 'coeffs'):
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        snap_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            snap_array[i] = self.get_snap(t_samples[i])

        return snap_array
