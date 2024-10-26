function Permu=simple_permutation(Ym,ratio)
Ym_Lr=size(Ym,1); Ym_Lc=size(Ym,2); Ym_L=Ym_Lr*Ym_Lc;
a=speye(Ym_L);
v=1:Ym_L;
vr=reshape(v,Ym_Lr,Ym_Lc);
vri=im2col(vr,[ratio,ratio],'distinct');
vrir=reshape(vri,1,[]);
Permu=a(:,vrir)';