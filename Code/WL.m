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



function [K,runtime] = WL(Graphs, h, nl)

    N = size(Graphs,2);
    Lists = cell(1, N);
    K = cell(1, h+1);
    n_nodes = 0;
    
    %for each graph 
    for i = 1:N
      %adjacency lists
      Lists{i} = Graphs(i).al; 
      %n_nodes (the total number of nodes in the dataset 
      n_nodes = n_nodes + size(Graphs(i).am,1);
    end
    
    %each column j of phi will be the explicit feature representation for the graph j
    phi = sparse(n_nodes, N); 

    % for measuring runtime
    t = cputime; 

    
    % initialize the node labels for each graph with their labels or with degrees (for unlabeled graphs).

    
    % label_lookup is an associative array, which will contain the
    % mapping from multiset labels (strings) to short labels (integers)
    label_lookup = containers.Map();
    label_counter = uint32(1);
    
    %if we want to use original node labels 
    if nl == 1
        %for each graph 
        for i = 1:N
            %for example, for graph 1, labels{1} is [23x1]
            labels{i} = zeros(size(Graphs(i).nl.values,1), 1, 'uint32');
            
            % for each node  
            for j = 1:length(Graphs(i).nl.values)
                str_label = num2str(Graphs(i).nl.values(j));
                if ~isKey(label_lookup, str_label)
                    %if the augmented labels is not contained in map 
                    label_lookup(str_label) = label_counter; %new key-value pair 
                    labels{i}(j) = label_counter;            %store counter in sparse matrix
                    label_counter = label_counter+1;         %update counter 
                else
                    labels{i}(j) = label_lookup(str_label); 
                end
                
                %phi: total # of nodes x 188
                phi(labels{i}(j), i) = phi(labels{i}(j),i) + 1;
                
            end
        end
    end
    
    
    L = label_counter-1;
    disp(['Number of original labels: ',num2str(L)]);
    clear Graphs;
    
    %inner product of kernels 
    K{1} = full(phi' * phi); %convert sparse matrix to full matrix
    
    iter = 1;
    new_labels = labels;
    
    while iter <= h

      disp(['iter=',num2str(iter)]);
      
      % create an empty map 
      label_lookup = containers.Map();
      label_counter = uint32(1);
      
      % create a sparse matrix for feature representations of graphs
      phi = sparse(n_nodes, N);
      
      %for each graph 
      for i = 1:N
          %for each node
          for v = 1:length(Lists{i})
              % form a multiset label of the node v of the i'th graph and convert it to a string
              long_label = [labels{i}(v), sort(labels{i}(Lists{i}{v}))'];
              long_label_2bytes = typecast(long_label, 'uint16');
              long_label_string = char(long_label_2bytes);
              
              % if the multiset label has not yet occurred, add it to the
              % lookup table and assign a number to it
              if ~isKey(label_lookup, long_label_string)
                label_lookup(long_label_string) = label_counter;
                new_labels{i}(v) = label_counter;
                label_counter = label_counter+1;
              else
                new_labels{i}(v) = label_lookup(long_label_string);
              end
          end
          
        % fill the column for i'th graph in phi
        aux = accumarray(new_labels{i}, ones(length(new_labels{i}),1));
        
        phi(new_labels{i}, i) = phi(new_labels{i}, i) + aux(new_labels{i});
      end
      
      L = label_counter - 1;
      disp(['Number of compressed labels: ',num2str(L)]);
      K{iter + 1} = K{iter} + full(phi' * phi);
      
      labels = new_labels;
      iter = iter+1;
      
    end
    runtime=cputime-t; % computation time of K
    
end

