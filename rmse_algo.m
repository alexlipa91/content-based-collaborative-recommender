function [ param ] = rmse_algo ( U_1, F, shr ) 
    z = U_1 == 0;
    
    [u,U_2] = compute_u(U_1);
    param.u = u;
    
    [b_u,U_3] = compute_bu(U_2, z,shr);
    param.b_u = b_u';
            
    clear U_2;
    
    U_p = U_1 .* (U_1 > 3.5);
    C = F*F';
    S = U_p * C;    
    [a,U_4] = compute_a(U_3,S,z);
    param.a = a;
    
    clear U_3;
    clear S;
    clear C;
    
    D = F'*F;
    T = U_p * F;
    V = T'*T;
    [b_k,U_5] = compute_bk(U_4,U_p,F,T,V,D,z);
    param.b_k = b_k';
    
    clear U_4;
    
    [x_u,y_k] = compute_cuk(U_5,F,D,T,3);
    param.x_u = x_u';
    param.y_k = y_k';    
end