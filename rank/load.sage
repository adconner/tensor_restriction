import subprocess

def tensor_rank_try1(T,r,ftol=1e-5,maxit=600,verbose=False):
    fout = sage.misc.temporary_file.tmp_filename()
    cmd = ['gen/rank/tensor_rank_try1',str(r),fout]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    a=len(T)
    b,c = T[0].dimensions()
    proc.stdin.write('%d %d %d\n' % (a,b,c))
    for m in T:
        for x in m.list():
            proc.stdin.write('%f\n' % CDF(x))
    proc.communicate()
    dec = open(fout).readlines()
    dec = [CDF(*map(float,e)) for e in groupn(dec,2)]
    r = len(dec) // (a+b+c)
    dec = groupn(dec,r)
    dec = [ (v[:a],v[a:a+b],v[a+b:]) for i in range(r) for v in [[es[i] for es in dec]]]
    out.append(dec)
    return dec

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
    cmd = ['gen/rank/tensor_rank',brank_out if find_brank else 'no',
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
        dec = open(brank_out).readlines()
        dec = [CDF(*map(float,e)) for e in groupn(dec,2)]
        r = len(dec) // (a+b+c)
        dec = groupn(dec,r)
        dec = [ (v[:a],v[a:a+b],v[a+b:]) for i in range(r) for v in [[es[i] for es in dec]]]
        out.append(dec)
    if find_rank:
        dec = open(rank_out).readlines()
        dec = [CDF(*map(float,e)) for e in groupn(dec,2)]
        r = len(dec) // (a+b+c)
        dec = groupn(dec,r)
        dec = [ (v[:a],v[a:a+b],v[a+b:]) for i in range(r) for v in [[es[i] for es in dec]]]
        out.append(dec)
    return out

# vim: ft=python
