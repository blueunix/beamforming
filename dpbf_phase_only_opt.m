%% dpbf_phase_only_opt.m
% Phase-only DPBF (H/V) optimization with small-amplitude taper option
% - 1D horizontal design for N = 8 antennas (per polarization)
% - Target: ±75° at -10 dB (i.e. -10 dB at phi0 = 75°)
% - Uses projected gradient descent on phase variables (w = exp(j*psi))
% - Features: adaptive step size, multiple random restarts, plotting
%
% Usage: run this script in MATLAB. It will produce figures and save results.

clearvars; close all; clc;

%% -------------------- Parameters --------------------
lambda = 1;                  % normalized wavelength
dx = 0.5 * lambda;           % spacing
N = 8;                       % number of elements in horizontal aperture
k = 2*pi / lambda;

% element positions, centered
n = (0:N-1).';
x = (n - (N-1)/2) * dx;      % N x 1

% angular grid for optimization and plots
phi_deg = -90:0.5:90;        % degrees (can change resolution)
phi = deg2rad(phi_deg);      % radians
M = numel(phi);

% Target envelope parameters
phi0_deg = 75;                           % ±phi0 is -10 dB point
phi0 = deg2rad(phi0_deg);
R = 10^(-10/20);                         % amplitude at ±phi0 (~0.31622777)

% Envelope: smooth (raised-cosine inside ±phi0, exponential outside)
alpha = 0.2;                             % outside decay rate (tunable)
E = zeros(M,1);
for i=1:M
    ph = abs(phi(i));
    if ph <= phi0
        E(i) = (1+R)/2 + (1-R)/2 * cos(pi * (ph/phi0));  % smooth from 1 -> R
    else
        E(i) = R * exp(-alpha * (ph - phi0));
    end
end
% Normalize E peak to 1 just in case
E = E / max(E);

% Steering matrix A (M x N) with A(i,n) = exp(-j k x_n sin(phi_i))
A = exp(-1j * k * ( sin(phi).' * x.' ));   % (M x N)  (note sin(phi) as column)

%% -------------------- Optimization settings --------------------
maxIter = 2000;            % max inner iterations per restart
restarts = 20;             % number of random restarts
tolJ = 1e-6;               % tolerance on objective change
mu0 = 1e-3;                % initial step size
mu_decay = 0.999;         % decay factor per iter (keeps step adaptively smaller)
use_alternating = false;   % if true: alternate H and V updates; else simultaneous
display_progress = true;   % show progress per restart

% angle weighting to emphasize main sector (optional)
w_ang = ones(M,1);
% emphasize region inside ±phi0 for fitting (e.g. weight 3 inside)
w_ang(abs(phi) <= phi0) = 3;

%% -------------------- Helper functions --------------------
% function to compute AF and residual and objective
compute_rJ = @(wH,wV) deal( A*wH, A*wV );  % will compute below in loop

%% -------------------- Multi-restart optimization --------------------
bestJ = Inf;
best_wH = [];
best_wV = [];
best_psiH = [];
best_psiV = [];

rng(0); % seed for reproducibility
for r = 1:restarts
    % initialize phases (random)
    psiH = 2*pi*rand(N,1) - pi;
    psiV = 2*pi*rand(N,1) - pi;
    wH = exp(1j * psiH);
    wV = exp(1j * psiV);

    mu = mu0;
    Jprev = Inf;

    for it = 1:maxIter
        % compute AFs and residuals
        AF_H = A * wH;              % M x 1
        AF_V = A * wV;              % M x 1

        % residual r = |AF_H|^2 + |AF_V|^2 - E^2
        r_vec = (abs(AF_H).^2 + abs(AF_V).^2) - (E.^2);
        % weighted objective
        J = sum( w_ang .* (r_vec.^2) );

        % check convergence
        if abs(Jprev - J) < tolJ
            break;
        end
        Jprev = J;

        % compute q vectors (N x 1)
        % qH = A' * ( conj(AF_H) .* (w_ang .* r_vec) )
        qH = A' * ( conj(AF_H) .* (w_ang .* r_vec) );   % N x 1
        qV = A' * ( conj(AF_V) .* (w_ang .* r_vec) );   % N x 1

        % gradients w.r.t. phases (N x 1 real)
        gH = 4 * imag( wH .* qH );   % as derived
        gV = 4 * imag( wV .* qV );

        % update phases (optionally alternate)
        if use_alternating
            % alternate updates H then V
            psiH = psiH - mu * gH;
            wH = exp(1j * psiH);
            % recompute AF_V? we keep AF_V until next iter; could recompute for more accuracy
            psiV = psiV - mu * gV;
            wV = exp(1j * psiV);
        else
            % simultaneous update
            psiH = psiH - mu * gH;
            psiV = psiV - mu * gV;
            wH = exp(1j * psiH);
            wV = exp(1j * psiV);
        end

        % adaptive step-size (decay)
        mu = mu * mu_decay;

        % optional: simple safeguard - if J grows a lot, reduce step and rollback
        if it>1 && J > 1.5 * Jprev
            % rollback and shrink mu
            psiH = psiH + mu * gH;   % rollback (add back previous step)
            psiV = psiV + mu * gV;
            wH = exp(1j * psiH);
            wV = exp(1j * psiV);
            mu = mu * 0.1;           % aggressively shrink
        end
    end % inner iterations

    if display_progress
        fprintf('Restart %2d/%d: it=%4d, J=%.6e, mu_final=%.3e\n', r, restarts, it, J, mu);
    end

    % store best
    if J < bestJ
        bestJ = J;
        best_wH = wH;
        best_wV = wV;
        best_psiH = psiH;
        best_psiV = psiV;
        best_r_vec = r_vec;
    end
end % restarts

fprintf('Best objective after %d restarts: J = %.6e\n', restarts, bestJ);

%% -------------------- Optional: apply a tiny amplitude taper --------------------
apply_taper = true;
if apply_taper
    % choose sigma_t so that power loss <= 0.5 dB (rough heuristic search)
    % a(x) = exp(-x^2/(2*sigma_t^2)); search sigma_t from large->small
    sigma_grid = linspace(0.1, 5, 200);  % in units of lambda
    orig_power = sum(abs(best_wH).^2) + sum(abs(best_wV).^2);
    chosen_sigma = sigma_grid(end);
    for s = sigma_grid
        a = exp( - (x.^2) / (2 * s^2) );
        p_after = sum( (a.^2) .* abs(best_wH).^2 ) + sum( (a.^2) .* abs(best_wV).^2 );
        loss_db = 10*log10(p_after / orig_power);
        if loss_db >= -0.5  % loss no more than 0.5 dB
            chosen_sigma = s;
            break;
        end
    end
    taper = exp( - (x.^2) / (2 * chosen_sigma^2) );
    W_H = (taper .* best_wH);   % complex weights
    W_V = (taper .* best_wV);
    fprintf('Applied Gaussian taper with sigma=%.3f lambda (loss ~ %.3f dB)\n', chosen_sigma, 10*log10((sum((taper.^2).*abs(best_wH).^2)+sum((taper.^2).*abs(best_wV).^2))/orig_power));
else
    W_H = best_wH;
    W_V = best_wV;
    taper = ones(N,1);
end

%% -------------------- Compute final patterns and plot --------------------
AF_H_final = A * W_H;   % M x 1
AF_V_final = A * W_V;
P_sum = abs(AF_H_final).^2 + abs(AF_V_final).^2;
P_sum_db = 10*log10( P_sum / max(P_sum) );

% Target in dB
E_db = 20*log10(E / max(E) + eps);

% Plot combined pattern
figure('Name','DPBF Combined Pattern','NumberTitle','off');
plot(phi_deg, P_sum_db, 'LineWidth', 1.5); hold on;
plot(phi_deg, E_db, '--','LineWidth',1.5);
grid on; xlabel('Azimuth (deg)'); ylabel('Normalized Power (dB)');
title(sprintf('Combined P_{sum}(\\phi) vs Target Envelope (N=%d)', N));
legend('P_{sum} (result)', 'Target (E, dB)', 'Location','SouthOutside');
ylim([-40 2]);

% Plot H and V individual patterns
figure('Name','Per-polarization Patterns','NumberTitle','off');
plot(phi_deg, 10*log10(abs(AF_H_final).^2 / max(abs(AF_H_final).^2)+eps), '-','LineWidth',1.2); hold on;
plot(phi_deg, 10*log10(abs(AF_V_final).^2 / max(abs(AF_V_final).^2)+eps), '-','LineWidth',1.2);
plot(phi_deg, E_db, '--','LineWidth',1.2);
grid on; xlabel('Azimuth (deg)'); ylabel('Normalized Power (dB)');
title('Per-polarization patterns (normalized)');
legend('H','V','Target','Location','SouthOutside');
ylim([-40 2]);

% Visualize amplitude taper and phases
figure('Name','Weights (amp & phase)','NumberTitle','off');
subplot(2,1,1);
stem((1:N), abs(W_H),'o-','LineWidth',1.2); hold on;
stem((1:N), abs(W_V),'x-','LineWidth',1.2);
xlabel('Element index'); ylabel('Amplitude'); title('Amplitude per element (H & V)'); grid on;
legend('H','V');
subplot(2,1,2);
plot((1:N), angle(W_H),'o-'); hold on;
plot((1:N), angle(W_V),'x-'); grid on;
xlabel('Element index'); ylabel('Phase (rad)'); title('Phase per element (H & V)');
legend('H','V');

% Print some performance metrics
% - max absolute dB error inside sector ±phi0
idx_in = abs(phi) <= phi0;
max_abs_err_db = max(abs(P_sum_db(idx_in) - E_db(idx_in)));
rmse_db = sqrt(mean((P_sum_db(idx_in) - E_db(idx_in)).^2));
fprintf('Max absolute error (dB) inside ±%.1f°: %.3f dB\n', rad2deg(phi0), max_abs_err_db);
fprintf('RMSE (dB) inside ±%.1f°: %.3f dB\n', rad2deg(phi0), rmse_db);

%% -------------------- Save results --------------------
save('dpbf_results_8elem.mat','W_H','W_V','best_psiH','best_psiV','taper','phi_deg','P_sum_db','E_db');

fprintf('Results saved to dpbf_results_8elem.mat\n');

%% End of script
