# This source code is intended to be executed in a SageMath Jupyter notebook environment.
# It utilizes specific features and functions available within SageMath that may not
# be compatible with standard Python or other programming environments.
# 
# To run this code:
# 1. Ensure you have SageMath installed on your system.
# 2. Open a Jupyter notebook using SageMath.
# 3. Copy the contents of this file into a cell in the Jupyter notebook.
# 4. Execute the cell to see the results.
#
# For more information about SageMath and how to set up a Jupyter notebook with SageMath,
# please visit: https://www.sagemath.org

import sqlite3, time, os


def Xd_element(us, ls):
    d = len(us)
    assert len(ls)==d
    assert all(l>=0 for l in ls)
    us = tuple(u%N for u in us)
    ls = tuple(ls)
    return FormalSum([ (1, us+ls) ])


def push(ls,i,r):
    d = len(ls)
    assert all(l>=0 for l in ls)
    assert 0<=i<=d-2
    assert r>=0
    assert ls[i]+ls[i+1]>=r
    return ls[:i] + (ls[i]+ls[i+1]-r,) + ls[i+2:]


def Diterpre(vs,ms,us,ls):
    e = len(vs)
    d = len(us)
    assert len(ms)==e
    assert len(ls)==d
    assert all( m>=0 for m in ms)
    assert all( l>=0 for l in ls)
    assert d>=1

    if d==1:
        return Xd_element( vs+us, ms+ls ) 

    ret = FormalSum([])
    
    # 1st term
    ret += Diterpre(vs+(us[0]-us[1],), ms+(ls[0],), us[1:], ls[1:]  )

    # 2nd term
    for i in (2..d-1):
        ii = i-1 # use for index access
        for r in (ls[ii]..ls[ii-1]+ls[ii]):
            new_vs = vs+(us[ii]-us[ii+1],)
            new_ms = ms + (r,)
            new_us = us[:i-1] + us[i:]
            new_ls = push(ls, i-2, r)
            ret += (-1)**(r-ls[ii]) * binomial(r, ls[ii]) \
                   * Diterpre(new_vs, new_ms, new_us, new_ls)
    # 3rd term
    for i in (1..d-1):
        ii = i-1 # use for index access
        for r in (ls[ii]..ls[ii]+ls[ii+1]):
            new_vs = vs+(us[ii+1]-us[ii],)
            new_ms = ms + (r,)
            new_us = us[:i] + us[i+1:]
            new_ls = push(ls, i-1, r)
            ret -= (-1)**ls[ii] * binomial(r, ls[ii]) \
                   * Diterpre(new_vs, new_ms, new_us, new_ls)

    # 4th term
    for r in (ls[-1]..ls[-2]+ls[-1]):
        new_vs = vs+(us[-1],)
        new_ms = ms + (r,)
        new_us = us[:-1]
        new_ls = push(ls, d-2, r)
        ret += (-1)**(r-ls[-1]) * binomial(r, ls[-1]) \
               * Diterpre(new_vs, new_ms, new_us, new_ls)
    return ret


def Diter(us, ls):
    d = len(us)
    assert len(ls)==d
    assert all( l>=0 for l in ls)
    return Diterpre(vs=(), ms=(), us=us, ls=ls)


def generate_all_us(N, depth):
    """
    return the list of all tuples of length 'depth' consisting of the integers between 0 and N-1.
    """
    assert depth>=0
    assert N>=1
    if depth==0:
        return [()]
    return [us + (u,) for us in generate_all_us(N, depth-1) for u in (0..N-1)]


def generate_all_ls(depth, weight):
    """
    return the list of all tuples of length 'depth' consisting of integers
    where the sum of the elements in each tuple equals weight - depth.
    """
    if depth==0:
        return [()] if weight==0 else []
    return [ls + (k-1,) for k in (1..weight) for ls in generate_all_ls(depth-1, weight-k) ]


def generator_of_ker_to_Y(depth, weight, generate_inverse_formula= False):
    def replace(seq, i, v):
        assert 0<=i<len(seq)
        return seq[:i] + (v,) + seq[i+1:]
    ret = []
    for us in generate_all_us(N, depth):
        for ls in generate_all_ls(depth, weight):
            if any(us[i]==0 and ls[i]==0 for i in range(depth)):
                ret.append( Xd_element(us, ls) )
            if generate_inverse_formula:
                for i in range(depth):
                    us_inv = us[:i] + (-us[i],) + us[i+1:]
                    ret.append( Xd_element(us,ls) \
                         - (-1)**ls[i]* Xd_element(replace(us, i, -us[i]), ls)   )
            for i in range(depth):
                for M in N.divisors():
                    if M!=1 and us[i]%M==0 and (us[i],ls[i])!=(0,0):
                        ret.append( Xd_element(us,ls) \
                         - sum( Xd_element(replace(us, i, us[i]//M+s*N//M), ls) for s in (0..M-1)) )
    return ret

def calc_dimension_of_the_space_quoted_by_vals(_N, depth, weight, vals, sparse = True):
    N = _N
    assert depth<=weight
    assert N>=1
    ind_from_us_ls = {}
    for us in generate_all_us(N, depth):
        for ls in generate_all_ls(depth, weight):
            if any( us[i]*2>N or ( us[i]*2%N==0 and ls[i]%2==1)  for i in range(depth)):
                continue
            ind_from_us_ls[us+ls] = len(ind_from_us_ls)

    def val_to_vector(val):
        ret = vector(ZZ, len(ind_from_us_ls), immutable=False, sparse = sparse)
        for coeff, us_ls in val:
            us,ls = list(us_ls[:depth]), list(us_ls[depth:])
            if any( us[i]*2%N==0 and ls[i]%2==1 for i in range(depth) ):
                continue
            for i in range(depth):
                if us[i]*2>N:
                    us[i] = N - us[i]
                    coeff *= (-1)**ls[i]
            us_ls = tuple(us+ls)
            ret[ind_from_us_ls[us_ls]] += coeff
        return ret

    mat = matrix([ val_to_vector(val) for val in vals ], sparse = sparse)
    return mat.ncols() - mat.rank()

def calc_dimension(_N, depth, weight, sparse = True):
    global N
    N = _N
    assert depth<=weight
    assert N>=1

    vals = []
    # image of D_d_iter
    for us in generate_all_us(N, depth):
        us2 = tuple( (-u)%N for u in us )
        if us2<us:
            continue
        for ls in generate_all_ls(depth, weight):
            vals.append( Diter(us, ls) )
    # generator of ker(Yd -> Xd)
    for val in generator_of_ker_to_Y(depth, weight, generate_inverse_formula= False):
        vals.append(val)
    return calc_dimension_of_the_space_quoted_by_vals(N, depth, weight, vals, sparse)

def make_sqlite_table():
    conn = sqlite3.connect('dimensions.db')
    cur = conn.cursor()
    
    # Create the table if it does not exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS dimension_data (
        N INTEGER,
        weight INTEGER,
        depth INTEGER,
        dim INTEGER,
        used_time REAL,
        PRIMARY KEY (N, weight, depth)
    )
    ''')
    conn.commit()

def calc_dimension_with_sqlite(N, depth, weight):
    """
    Calculate the dimension for given N, depth, and weight using SQLite3 database.
    If the data is already in the database, return the stored dimension.
    Otherwise, calculate the dimension, store it in the database, and return it.
    """

    make_sqlite_table()
    conn = sqlite3.connect('dimensions.db')
    cur = conn.cursor()

    # Search for existing data in the database
    cur.execute('''
    SELECT dim FROM dimension_data WHERE N = ? AND weight = ? AND depth = ?
    ''', (int(N), int(weight), int(depth)))
    
    row = cur.fetchone()

    if row:
        # If data exists, return the stored dimension
        conn.close()
        return row[0]
    else:
        # If data does not exist, calculate the dimension and update the database
        start_time = time.time()
        dim = calc_dimension(N, depth, weight)
        used_time = time.time() - start_time

        cur.execute('''
        INSERT INTO dimension_data (N, weight, depth, dim, used_time)
        VALUES (?, ?, ?, ?, ?)
        ''', (int(N), int(weight), int(depth), int(dim), used_time))
        
        conn.commit()
        conn.close()
        return dim
    
def test_Ydim():
    def calc_dim_by_formula(N, depth, weight):
        assert N>=1
        if depth==1:
            if weight<=0:
                return 0
            if N==1:
                return 1 if weight>=3 and weight%2==1 else 0
            elif N==2:
                return 1 if weight%2==1 else 0
            else:
                if weight==1:
                    return euler_phi(N)//2 + len(N.prime_divisors()) - 1
                else:
                    return euler_phi(N)//2
        return sum( calc_dim_by_formula(N,1,w) * calc_dim_by_formula(N, depth-1, weight-w) for w in (1..weight-1))
    global N
    for depth in (1..8):
        for weight in (depth..20):
            for _N in (1..500):
                N = _N
                if len(generate_all_us(N, depth)) * len(generate_all_ls(depth, weight) )>=100:
                    break
                
                vals = generator_of_ker_to_Y(depth, weight, generate_inverse_formula= True)
                dim1 = calc_dimension_of_the_space_quoted_by_vals(_N=N, depth=depth, weight=weight, vals=vals)
                dim2 = calc_dim_by_formula(N = N, depth = depth, weight = weight)
                print(f"{depth=}, {weight=}, {N=}, {dim1=}, {dim2=}, {dim1==dim2}")
                assert dim1==dim2                

def test():
    for N in (200..3000):
        dim = calc_dimension_with_sqlite(N=N, depth = 2, weight=2)
        print(N, dim)

def test2():
    for N in (200..3000):
        import time
        start = time.time()
        dim_sparse = calc_dimension(_N=N, depth = 2, weight=2, sparse = True)
        print(time.time()-start)

        print(N, dim_sparse)

def main_parallel():
    @parallel(ncpus = 8)
    def f(N, depth, weight):
        dim = calc_dimension_with_sqlite(N, depth, weight)
        print(f"{depth=}, {weight=}, {N=}, {dim=}")
        return dim
    
    ins = [(N,2,2) for N in (1..100)]
    res = list(f(ins))

def main():
    for N in (1..100):
        depth = 2
        weight = 2
        dim = calc_dimension_with_sqlite(N = N, depth = depth, weight = weight)
        print(f"{depth=}, {weight=}, {N=}, {dim=}")

main()
print("Finish!")
