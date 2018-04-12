% Compute h-step Weisfeiler-Lehman kernel for a set of graphs
% Input:    Graphs - a 1xN array of graphs
%           Graphs(i).am is the adjacency matrix of the i'th graph, 
%           Graphs(i).al is the adjacency list of the i'th graph, 
%           Graphs(i).nl.values is a column vector of node labels for the i'th graph.
%         
%           (Graphs(i) may have other fields, but they will not be used here)
%
%           h - a natural number: number of iterations of WL
%           nl - a boolean: 1 if we want to use original node labels, 0 otherwise
%
% Output:   K - a h+1-element cell array of NxN kernel matrices K for each iter = 0,...,h
%           runtime - scalar (total runtime in seconds)

function [K,runtime] = WLedge(Graphs,h,nl)

N=size(Graphs,2);
Lists = cell(1,N);
K = cell(1,h+1);
n_nodes=0;
% compute adjacency lists and n_nodes, the total number of nodes in the dataset
for i=1:N
  Lists{i}=Graphs(i).al;
  n_nodes=n_nodes+size(Graphs(i).am,1);
end
% copy adjacency matrices
for i=1:N
  AMs{i}=Graphs(i).am;
end

t=cputime; % for measuring runtime

%%% INITIALISATION
% initialize the node labels for each graph with their labels or 
% with degrees (for unlabeled graphs).

label_lookup=containers.Map();
label_counter=uint32(1);

% label_lookup is an associative array, which will contain the
% mapping from multiset labels (strings) to short labels (integers)

if nl == 1
    %for each graph
    for i = 1:N
        
        % the type of labels{i} is uint32, meaning that it can only handle
        % 2^32 labels and compressed labels over all iterations. If
        % more is needed, switching (all occurences of uint32) to
        % uint64 is a possibility
        labels{i} = zeros(size(Graphs(i).nl.values,1),1,'uint32');
    
        %for each node  
        for j = 1:length(Graphs(i).nl.values)
            str_label = num2str(Graphs(i).nl.values(j));
            if ~isKey(label_lookup, str_label)
                label_lookup(str_label) = label_counter;
                labels{i}(j) = label_counter;
                label_counter = label_counter + 1;
            else
                labels{i}(j) = label_lookup(str_label);
            end
        end
    end
end


L = double(label_counter)-1;
clear Graphs;
disp(['Number of original labels: ', num2str(L)]);
disp(['Number of potential edge features: ', num2str(L*(L+1)/2)]);
ed = sparse(L*(L+1)/2, N);

% test 
l = labels{i}(1:3);
l(2) = 2
l(3) = 3
labels_aux0 = repmat(l, 1, length(l));
a0 = min(labels_aux0, labels_aux0');
b0 = max(labels_aux0, labels_aux0');
% test 


%for each graph
for i = 1:N
    %repmat: repeat copies of array  
    labels_aux = repmat(double(labels{i}), 1, length(labels{i}));
    a = min(labels_aux, labels_aux');
    b = max(labels_aux, labels_aux');
    I = triu(AMs{i} ~= 0, 1);
  
    %.*: array multiplication 
    Ind = (a(I) - 1) .* (2*L + 2 - a(I))/2 + b(I) - a(I) + 1;
  
    diff = max(Ind) - min(Ind);
    aux = accumarray(Ind, ones(nnz(I),1), [], [], [], (min(Ind) > 5000 || diff > 3000));
  
    % sparse of full accumarray depending on the range of values in Ind
    % (and based on my empirical observations on the speed of accumarray)
    ed(Ind,i) = aux(Ind);
end

ed = ed(sum(ed,2)~=0, :);
K{1} = full(ed' * ed);


%%% MAIN LOOP
iter=1;
new_labels=labels;
while iter<=h
  disp(['iter=',num2str(iter)]);
  % create an empty lookup table
  label_lookup=containers.Map();
  label_counter=uint32(1);
  for i=1:N
    for v=1:length(Lists{i})
      % form a multiset label of the node v of the i'th graph
      % and convert it to a string
      long_label=[labels{i}(v), sort(labels{i}(Lists{i}{v}))'];
      long_label_2bytes=typecast(long_label,'uint16');
      long_label_string=char(long_label_2bytes);
      % if the multiset label has not yet occurred, add it to the
      % lookup table and assign a number to it
      if ~isKey(label_lookup, long_label_string)
        label_lookup(long_label_string)=label_counter;
        new_labels{i}(v)=label_counter;
        label_counter=label_counter+1;
      else
        new_labels{i}(v)=label_lookup(long_label_string);
      end
    end
  end
  
  L=double(label_counter)-1;
  disp(['Number of compressed labels: ',num2str(L)]);
  disp(['Number of potential edge features: ',num2str(L*(L+1)/2)]);
  labels=new_labels;
  ed=sparse(L*(L+1)/2,N);
  for i=1:N
    labels_aux=repmat(double(labels{i}),1,length(labels{i}));
    a=min(labels_aux, labels_aux');
    b=max(labels_aux, labels_aux');
    I=triu(AMs{i}~=0,1);
    Ind=(a(I)-1).*(2*L+2-a(I))/2+b(I)-a(I)+1;
    minind=min(Ind);
    diff = max(Ind)-minind;
    aux=accumarray(Ind,ones(nnz(I),1),[],[],[],(minind > 5000 || diff > 3000));
    % sparse of full accumarray depending on the range of values in Ind
    % (and based on my empirical observations on the speed of accumarray)
    ed(Ind,i)=aux(Ind);
  end
  ed=ed(sum(ed,2)~=0,:);
  K{iter+1}=K{iter}+full(ed'*ed);
  iter=iter+1;
end
runtime=cputime-t; % computation time of K
end

