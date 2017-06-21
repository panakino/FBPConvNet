function [out]=dev_fun(img)

img_dx=padarray(diff(img,[],2),[0 1],'post');
img_dy=padarray(diff(img,[],1),[1 0],'post');
img_hes=padarray(diff(diff(img,[],1),[],2),[1 1],'post');

img_dx2=padarray(diff(img_dx,[],2),[0 1],'post');
img_dy2=padarray(diff(img_dy,[],1),[1 0],'post');
img_hes2=padarray(diff(diff(img_hes,[],1),[],2),[1 1],'post');

out=cat(3,img,img_dx,img_dy,img_hes,img_dx2,img_dy2,img_hes2);