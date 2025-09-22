import pickle
from pathlib import Path
from ipdb import set_trace


if __name__ == "__main__":
    # get directory of this script
    current_dir = Path(__file__).parent

    # load correspondences
    Corrs_dir = current_dir / "CorrsDict.pkl"
    with open(Corrs_dir, 'rb') as f:
        CorrsDict = pickle.load(f)

    query_list = list(CorrsDict.keys())
    
    # load RANSAC results threshold 10.0 px
    Results_dir = current_dir / "RANSACresults_10.0_dict.pkl"
    with open(Results_dir, 'rb') as f:
        Results = pickle.load(f)

    query_result = list(Results.values())[0]
    methods = list(query_result.keys())
    print(methods[0])
    print('focal length error:', query_result[methods[0]]['f_err'], 'px')
    print('runtime:', query_result[methods[0]]['runtime'], 'ms')
    print('iterations:', query_result[methods[0]]['info']['iterations'])
    print(methods[3])
    print('focal length error:', query_result[methods[3]]['f_err'], 'px')
    print('runtime:', query_result[methods[3]]['runtime'], 'ms')
    print('iterations:', query_result[methods[3]]['info']['iterations'])
