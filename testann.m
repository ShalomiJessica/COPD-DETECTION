function err=testann(ann, test_xx, test_yy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 ann = ffann(ann, test_xx);
 
 if ann.layers{ann.no_of_layers}.type ~= 'f'
  zz=[];
  for k=1:ann.layers{ann.no_of_layers}.no_featuremaps
                   ss =size(ann.layers{ann.no_of_layers}.featuremaps{k});
                   zz =[zz; reshape(ann.layers{ann.no_of_layers}.featuremaps{k}, ss(1)*ss(2), ss(3))];
  end
   ann.layers{ann.no_of_layers}.outputs = zz;
 end
 
[a, l1]=max(ann.layers{ann.no_of_layers}.outputs, [],1);
[b, l2]=max(test_yy, [], 1);
idx = find(l1 ~= l2);

err = length(idx)/prod(size(l1));

display 'test error is'
err
 
