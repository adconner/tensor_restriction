import subprocess

def tensor_rank_try1(T,r,ftol=1e-5,maxit=600,verbose=False):
    cmd = ['gen/restrict/rank/tensor_rank_try1',str(r)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    a=len(T)
    b,c = T[0].dimensions()
    proc.stdin.write('%d %d %d\n' % (a,b,c))
    for m in T:
        for x in m.list():
            proc.stdin.write('%f\n' % CDF(x))
    out = []
    for i in range(r):
        cur = [CDF(*map(float,proc.stderr.readline().split()))
            for j in range(a+b+c)]
        out.append((cur[:a],cur[a:a+b],cur[a+b:]))
    return out

def tensor_rank(T,find_brank=True,find_rank=True,brlower=0,brupper=None):
    if brlower == 0:
        brlower = ceil(summarize(T,pmax=2))
    if brupper is None:
        brupper = 1e9
        for _ in range(3):
            brupper = min(brupper,sum([m.rank() for m in T]))
            T = tensor_cycl(T)
    if find_brank:
        brank_out = sage.misc.temporary_file.tmp_filename()
    if find_rank:
        rank_out = sage.misc.temporary_file.tmp_filename()
    cmd = ['rank/tensor_rank',brank_out if find_brank else 'no',
            rank_out if find_rank else 'no',str(brlower),str(brupper)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    a=len(T)
    b,c = T[0].dimensions()
    proc.stdin.write('%d %d %d\n' % (a,b,c))
    for m in T:
        for x in m.minors(1):
            proc.stdin.write('%f\n' % CDF(x))
    proc.communicate()
    out=[]
    if find_brank:
        br = open(brank_out).read().split()
        br = [CDF(*map(float,e)) for e in groupn(br,2)]
        br = [(v[:a],v[a:a+b],v[a+b:]) for v in groupn(br,a+b+c)]
        out.append(br)
    if find_rank:
        r = open(rank_out).read().split()
        r = [CDF(*map(float,e)) for e in groupn(r,2)]
        r = [(v[:a],v[a:a+b],v[a+b:]) for v in groupn(r,a+b+c)]
        out.append(r)
    return out

# vim: ft=python
