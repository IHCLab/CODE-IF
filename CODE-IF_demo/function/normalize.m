function [out,MAX,MIN]=normalize(in)
in=double(in);
MAX=max(max(max(in)));
MIN=min(min(min(in)));
out= (in-MIN)/(MAX-MIN);