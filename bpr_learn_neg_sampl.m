% urm implicita ( controllare prima di chiamare )
% reg = [reg_a,reg_bk,reg_xu,reg_yk]
function[delta_st,par_st,par_mag_st,rec] = bpr_learn_neg_sampl( urm, icm, n_it, alpha, reg, test_pos )    
    %%%
    delta_st = zeros(1,n_it);
    par_st = cell(1,n_it);
    par_mag_st = cell(1,n_it);
    %%%
    
    param.u = nan;
	param.b_u = nan;
	param.a = rand();
	param.b_k = rand(1,size(icm,2));
	param.x_u = rand(1,size(urm,1));
	param.y_k = rand(1,size(icm,2));
    
    iter = 1;
    
    while(not(iter-1==n_it))
        u = datasample(1:size(urm,1),1);
        if( sum( urm(u,:) ) == 0 ) 
            continue;
        end
        i = datasample(find(urm(u,:)==1),1);
        j = datasample(find(urm(u,:)==0),1);        
        
        %%%% function(par) at (u,i,j) 
        [err,x_uij] = function_at( u, i, j, urm, icm, param );
        
        if ( x_uij > 0 )
            continue;
        end
                
        %%%% gradient(par) at (u,i,j)
        gr = compute_gradients( urm(u,:), icm, i, j, param.y_k, param.x_u(u) );
        
        %%%% new paramters
        par_new = update_param(u,x_uij,gr,alpha,param,reg);
        
        %%%% function(par_new) at (u,i,j)
        [err_new,x_uij_new] = function_at( u, i, j, urm, icm, par_new ); 

        param = par_new;        
        %%%
        
        delta_st(iter) = err_new-err;
        if (isnan(delta_st(iter)))
            delta_st(iter)=0;
        end
        par_st{iter} = par_new;
        par_mag_st{iter} = [norm(par_new.a),norm(par_new.b_k),norm(par_new.x_u),norm(par_new.y_k)];
        %%%        
        
        iter = iter + 1;
    end       
    
    recalls = recall( predictor(urm,icm,par_st{n_it},true), test_pos );
    rec = recalls(1:10);
end

function [err,x_uij] = function_at( u, i, j, urm, icm, par ) 
    w_k = (par.x_u(u) .* par.y_k) + par.b_k + par.a;
    w_k_i_j = w_k .* (icm(i,:)-icm(j,:));
    x_uij = urm(u,:) * (icm * w_k_i_j');
    err = -log(1/(1+exp(-x_uij)));
end

function [gr] = compute_gradients( u_prof, icm, i, j, y_k, x_u )
    icm_mod = icm .* repmat( (icm(i,:) - icm(j,:)), [size(icm,1),1] );
    icm_mod_2 = icm_mod .* repmat(y_k,[size(icm,1),1]);
    
    gr.a = u_prof * sum(icm_mod,2);
    gr.b_k = u_prof * icm_mod;
    gr.x_u = u_prof * sum(icm_mod_2,2);
    gr.y_k = x_u .* gr.b_k;    
end

function [param_new] = update_param(u,x_uij,gr,alpha,param,reg)
    term = @(x)1/(1+exp(x));
        
    param_new.x_u = param.x_u;
    
    a_step = - term(x_uij) * gr.a;
    b_k_step = - term(x_uij) * gr.b_k;
    x_u_step = - term(x_uij) * gr.x_u;
    y_k_step = - term(x_uij) * gr.y_k;
    
    if ( reg(1) > 0 )
        a_step = a_step + reg(1) * param.a;
    end
    if ( reg(2) > 0 )
        b_k_step = b_k_step + reg(2) * param.b_k;
    end
    if ( reg(3) > 0 )
        x_u_step = x_u_step + reg(3) * param.x_u(u);
    end
    if ( reg(4) > 0 )
        y_k_step = y_k_step + reg(4) * param.y_k;
    end
    
    param_new.a      = param.a      -alpha * a_step;
    param_new.b_k    = param.b_k    -alpha * b_k_step;
    param_new.x_u(u) = param.x_u(u) -alpha * x_u_step;
    param_new.y_k    = param.y_k    -alpha * y_k_step;
    
    param_new.u = nan;
    param_new.b_u = nan;
end
