import matplotlib.pyplot as plt
import numpy as np

def get_data(ql_result, nl_result):
    
    m = {}
    for i in range(len(ql_result)):
        q = ql_result[i]
        n = nl_result[i]

        if q[0] not in m:
            m[q[0]] = {"q": [], "n": []}
        
        m[q[0]]["q"].append(q[2])
        m[q[0]]["n"].append(n[2])
    return m



k_array= [100, 1000, 5000, 10000, 50000]

page_size = [64, 128, 256, 512]

# nl_result = [(100, 64, 0.9502370357513428, 100.0), (100, 128, 0.831218957901001, 100.0), (100, 256, 0.7546070416768392, 100.0), (100, 512, 1.0194354057312012, 100.0), (1000, 64, 5.118876616160075, 1000.0), (1000, 128, 2.5726316769917807, 1000.0), (1000, 256, 1.3588182926177979, 1000.0), (1000, 512, 1.349292278289795, 1000.0), (5000, 64, 26.26106373469035, 5000.0), (5000, 128, 13.474780877431234, 5000.0), (5000, 256, 7.068738063176473, 5000.0), (5000, 512, 5.7744646072387695, 5000.0), (10000, 64, 53.76997454961141, 10000.0), (10000, 128, 27.468515316645306, 10000.0), (10000, 256, 14.537367582321167, 10000.0), (10000, 512, 10.781949679056803, 10000.0), (50000, 64, 270.58088358243305, 50022.0), (50000, 128, 144.36488167444864, 50000.0), (50000, 256, 82.01689728101094, 50000.0), (50000, 512, 74.3543180624644, 50689.0)]
# ql_result = [(100, 64, 0.9190916220347086, 100.0), (100, 128, 0.8123086293538412, 100.0), (100, 256, 0.7660703659057617, 100.0), (100, 512, 1.0144333044687908, 100.0), (1000, 64, 5.193581899007161, 1000.0), (1000, 128, 2.633220354715983, 1000.0), (1000, 256, 1.403469721476237, 1004.3333333333334), (1000, 512, 1.3556983470916748, 1000.0), (5000, 64, 26.523373047510784, 5000.0), (5000, 128, 13.385950326919556, 5000.0), (5000, 256, 6.7455776532491045, 5000.0), (5000, 512, 5.784783363342285, 5000.0), (10000, 64, 52.57084393501282, 10000.0), (10000, 128, 26.735239505767822, 10000.0), (10000, 256, 13.15861407915751, 10042.0), (10000, 512, 9.875536998112997, 10110.0), (50000, 64, 261.6442720095317, 50000.0), (50000, 128, 133.80324498812357, 50000.0), (50000, 256, 71.73548452059428, 50150.333333333336), (50000, 512, 53.87964026133219, 50361.666666666664)]


ql_result = [(100, 64, 0.95836075146993, 100.0), (100, 128, 0.8337086041768392, 100.0), (100, 256, 0.7766193548838297, 100.0), (100, 512, 0.8265613714853922, 102.33333333333333), (1000, 64, 5.58425235748291, 1000.0), (1000, 128, 2.695770581563314, 1000.0), (1000, 256, 1.3984344005584717, 1000.0), (1000, 512, 1.6751226584116619, 1000.0), (5000, 64, 26.695534626642864, 5000.0), (5000, 128, 13.369556665420532, 5000.0), (5000, 256, 6.613195339838664, 5000.0), (5000, 512, 5.020862976710002, 5154.666666666667), (10000, 64, 53.07685375213623, 10000.0), (10000, 128, 26.351319948832195, 10000.0), (10000, 256, 12.85532832145691, 10142.333333333334), (10000, 512, 10.946364084879557, 10373.666666666666), (50000, 64, 273.0872576236725, 50000.0), (50000, 128, 135.89197969436646, 50051.666666666664), (50000, 256, 64.28217840194702, 50272.666666666664), (50000, 512, 81.37850427627563, 50705.333333333336)]
nl_result = [(100, 64, 0.9476259549458822, 100.0), (100, 128, 0.8566019535064697, 100.0), (100, 256, 0.7920990784962972, 100.0), (100, 512, 1.0375640392303467, 100.0), (1000, 64, 5.214632987976074, 1000.0), (1000, 128, 2.6504762967427573, 1000.0), (1000, 256, 1.3870030244191487, 1000.0), (1000, 512, 1.092275063196818, 1000.0), (5000, 64, 26.78702433904012, 5000.0), (5000, 128, 13.805720647176107, 5000.0), (5000, 256, 7.747048695882161, 5000.0), (5000, 512, 5.766619046529134, 5000.0), (10000, 64, 55.3203980922699, 10000.0), (10000, 128, 28.813516934712727, 10000.0), (10000, 256, 15.736921946207682, 10000.0), (10000, 512, 12.693902015686035, 10000.0), (50000, 64, 288.218256632487, 50065.0), (50000, 128, 159.59022363026938, 50000.0), (50000, 256, 97.0799130598704, 50181.0), (50000, 512, 102.60742902755737, 50724.0)]
m = get_data(ql_result, nl_result)


def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge")   
    plt.xticks(_X, X)

fig, axs = plt.subplots(1, 5)
for i in range(len(k_array)):
    plt.figure("k : " + str(k_array[i]))

    subcategorybar(page_size, [m[k_array[i]]["n"], m[k_array[i]]["q"]])

plt.show()

