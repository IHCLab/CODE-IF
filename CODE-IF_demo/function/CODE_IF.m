function  [Z_fused]= CODE_IF(Yh,Ym,D,Z_DE,Permu,K,r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Code for paper CODE-IF [1]
%
%  [1] C.-H. Lin, C.-Y. Hsieh, and J.-T. Lin, "CODE-IF: A Convex/Deep Image Fusion Algorithm for
%      Efficient Hyperspectral Super-Resolution," IEEE Transactions on Geoscience and Remote Sensing,
%      vol. 62, pp. 1-18, 2024.
%
%  Input:    Yh    : the low-resolution hyperspectral image (H x W x HSI band)
%            Ym    : the high-resolution multispectral image (H x W x MSI band)
%            D     : the spectral response matrix (MSI band x HSI band)
%            Z_DE  : the rough network solution (H x W x HSI band)
%            Permu : the permutation matrix
%            K     : the point spread function (PSF)
%            r     : sampling factor (upsample ratio)
%
%  Output:   Z_fused : Fusion result (high-spatial resolution hyperspectral image)
%
%  To solve the following criterion:
%  ||Ym-DZ||_2^F + ||Ym-ZB||_2^F + λ/2||Z-Z_DE||_2^Q
%
%  Last modify : 2024 / 04 / 25
%  By Jhao-Ting Lin, Institute of Computer and Communication Engineering and the Department of Electrical Engineering,
%  National Cheng Kung University, Tainan, Taiwan (e-mail: q38091534@gs.ncku.edu.tw)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameter Setting
lambda = 0.05;    % value of λ (0.1~0.001) (depends on Z_DE quality)
p = 0.1;          % penalty parameter p in ADMM (1~0.01)
N = 15;           % model order N
iteration = 15;   % iteration number of ADMM

tic;
%% Get Image Size and Reshape
[~,~,Bh] = size(Yh);
[Rm,Cm,Bm] = size(Ym);
Yh3D=Yh;
Yh=reshape(Yh,[],Bh)';  % 3D to 2D
Ym=reshape(Ym,[],Bm)';  % 3D to 2D
g=reshape(K,[],1);

%% Permutate the Data
Ym = Ym*Permu';
Z_DE = reshape(Z_DE,[],size(Z_DE,3))'*Permu';

%% Get Basis E and S_DE
[E] = PCA(Yh3D,N);      % do PCA from Yh (instead of Z_DE)
sDE = E\Z_DE;

%% Alternating Direction Method of Multipliers (ADMM)
[S] = ADMM(N,Yh,Ym,E,D,g,lambda,p,sDE,iteration,r);

%% Reshape and Unpermute for Final Output
Z_fused = E*S*Permu;                        % unpermute
Z_fused = reshape(Z_fused',Rm,Cm,Bh);       % 2D to 3D
end

%% PCA (Principal Components Analysis)
function [E] = PCA(x_3D,N)
U = reshape(x_3D,[],size(x_3D,3))';
[M,~] = size(U);
[eV,~] = eig(U*U');
E = eV(:,M-N+1:end);
end

function [S] = ADMM(N,Yh,Ym,E,D,g,lambda,p,S_DE_2D,iteration,r)
%% Compute Some Matrixes for Fast Inplementation
Cb = [kron(g',E)',kron(speye(size(g,1)),D*E)']'; % C_bar
CbTCb = Cb'*Cb;
Cinv = CbTCb+(p/2)*speye(size(CbTCb));
ym = reshape(Ym,r*r*size(Ym,1),[]);
s_Cb = Cinv\kron(g',E)'*Yh+Cinv\(kron(speye(size(g,1)),D*E))'*ym;

%% Initialize Matrices
L = size(Ym,2);
s_DE = reshape(S_DE_2D,[],1);
s = s_DE;                    % warm start using S_DE
v = sparse(zeros(N*L,1));

%% Iteration
for i = 1:iteration
    % T update
    t = (lambda*s_DE)/(lambda+p)+p*(s-v)/(lambda+p);

    % S update
    tPv = (p/2)*(t+v);
    s = reshape( (Cinv\reshape(tPv,size(Cinv,2),[])+s_Cb), [], 1);

    % V update
    v = v-s+t;
end

S=reshape(s,N,L);
end
