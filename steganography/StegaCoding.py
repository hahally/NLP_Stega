import math

def  ADG(voc_p, u=None):
    sorted_dict= sorted(voc_p, key=lambda x: x[1], reverse = True) # [(w1,p1),...]
    v = [x[0] for x in sorted_dict]
    p = [x[1] for x in sorted_dict]
    
    p_max = p[1]
    first_token = v[0]
    if not u:
        u = 2**(math.floor(-math.log2(p_max)))
        k = math.floor(-math.log2(p_max))
    mean = 1./u
    group = {}
    for i in range(1,u):
        first = sorted_dict.pop(0)
        first_token = first[0]
        first_p = first[1]
        G = [first_token]
        p_g = [first_p]
        while sum(p_g) < mean:
            e = mean - sum(p_g)
            rest_p = [x[1] for x in sorted_dict]
            abs_dis = list(map(lambda x: abs(x-e), rest_p))
            idx = abs_dis.index(max(abs_dis))
            nearest = sorted_dict[idx]
            nearest_token = nearest[0]
            nearest_p = nearest[1]
            if nearest_p - e < e:
                G.append(nearest_token)
                sorted_dict.pop(idx)
                p_g.append(nearest_p)
            else:
                break
            
        rest_p = [x[1] for x in sorted_dict]
        mean = sum(rest_p)/(u-i)
        
        group[f'group_{i-1}'] = G
        
    v = [x[0] for x in sorted_dict]
    group[f'group_{u-1}'] = v
    return group, u, k
