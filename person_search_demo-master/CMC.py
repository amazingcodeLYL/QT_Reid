from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from collections import defaultdict

try:
    from .rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (np.asarray(g_pids)[order] == q_pid) & (np.asarray(g_camids)[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

'''
输入：
distmat：距离矩阵
q_pids：query集person id
g_pids：gallery集person id
q_camids：query集camera id
g_camids：gallery集camera id
max_rank：rank-n中n最大值

输出：
all_cmc：cmc曲线中rank-n的列表
mAP：所有符合要求的query数据的mAP
'''
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape # 距离矩阵的shape

	# 若gallery数据少于max_rank，将max_rank设为gallery数据集的数量
    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

	'''
	np.argsort()指定按照行进行排序，并返回每行按升序排列的元素下标
	举例：列表[1,2,0]，按行排列后，返回[2,0,1]
	'''
    indices = np.argsort(distmat, axis=1) # 每行按照距离大小进行排序，并获取排序后的下标顺序
 
 	'''
 	np.asarray(g_pids)[np.asarray(indices)]:与distmat距离矩阵相同规模的矩阵，但矩阵元素是按距离大小升序排列后对应的gallery的person id。举例：距离矩阵某一行为[1,0,5,6]，按行升序排列后得到的下标列表为[1,0,2,3]，gallery行人ID为[4,5,6,8]，则可以计算得到g_pids[indices]对应的那一行为[5,4,6,8]。
	np.asarray(q_pids)[:, np.newaxis]：将q_pids矩阵增加了一维。
	==：将g_pids[indices]与q_pids进行匹配，对应位置相同则元素为1，否则为0，则获得一个对应关系矩阵matches，该矩阵与距离矩阵规模相同。
	matches矩阵第i行第j个元素代表query第i个行人ID与gallery中与其距离第j近的数据行人ID是否相同。举例，matches[1][3]=1，说明query中第1个行人与距离第三近的gallery数据属于同一行人。
 	'''
 	# 进行ID匹配，计算匹配矩阵matched，便于计算cmc与AP
    matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = [] # 记录每个query数据的cmc数据
    all_AP = [] # 记录每个query数据的AP
    num_valid_q = 0. # 记录符合CMC与mAP计算的query数据的总数，便于计算总Rank-N

    for q_idx in range(num_q): # 对于query集合中的每一个数据
       
        q_pid = q_pids[q_idx] # 获取该数据的person id
        q_camid = q_camids[q_idx] # 获取该数据的camera id
        order = indices[q_idx] # 获取该数据相关的gallery数据距离排序

        # 删除与该query数据有相同person id ,camera id 的数据，相同摄像机相同行人的gallery数据不符合跨摄像机的要求
        remove = (np.asarray(g_pids)[order] == q_pid) & (np.asarray(g_camids)[order] == q_camid)
        keep = np.invert(remove) # 对remove进行翻转得到可以保留的数据的bool类型列表

        raw_cmc = matches[q_idx][keep] # 匹配矩阵只保留对应keep中为True的元素，得到该query数据的匹配列表
        if not np.any(raw_cmc):
             # 如果该query数据未在可以保留的gallery集中出现，说明该query数据不符合CMC与mAP计算要求，返回循环头
            continue

		'''
		计算每个query的cmc
		'''
        cmc = raw_cmc.cumsum() # 计算匹配列表的叠加和
        cmc[cmc > 1] = 1 # 根据叠加和得到该query数据关于gallery数据的Rank-N
        # 将该query数据的CMC数据加入all_AP列表便于之后计算mAP，可以通过指定max_rank来指定一行保留多少列，默认50列
        all_cmc.append(cmc[:max_rank]) 

		# 统计符合CMC与mAP计算的query数据的总数，便于计算总Rank-N
        num_valid_q += 1.

        '''
        计算每个query的AP
        '''
        num_rel = raw_cmc.sum() # 每个query数据的正确匹配总数
        tmp_cmc = raw_cmc.cumsum() # 计算匹配列表的叠加和
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)] # 计算每次正确匹配的准确率
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc # 将错误匹配的准确率降为0
        AP = tmp_cmc.sum() / num_rel # 计算平均准确度
        all_AP.append(AP) # 将该query数据的AP加入all_AP列表便于之后计算mAP

	# 如果符合CMC计算的query数据的总数小于等于0，则报错所有query数据都不符合要求
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32) # 将all_cmc转换为np.array类型
    # 将所有符合条件的query数据的Rank-N按列求和并取平均数，即可计算总CMC曲线中的Rank-N
    all_cmc = all_cmc.sum(0) / num_valid_q 
    # 平均准确率均值就是所有符合条件的query数据平均准确率的平均数
    mAP = np.mean(all_AP) 

    return all_cmc, mAP


def evaluate_py(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )
    else:
        return eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )


def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    use_metric_cuhk03=False,
    use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
    else:
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
