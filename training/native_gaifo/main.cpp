// RLGymPPO_CPP-inspired batched native collection + LibTorch PPO, extended for GAIfO.
// Models are exported from Goddard; all rollout, discriminator, GAE and PPO work is C++.
#include "environment.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

using torch::Tensor;
using torch::indexing::Slice;
using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void Sync(torch::Device device) { if (device.is_cuda()) torch::cuda::synchronize(); }
std::vector<Tensor> Parameters(torch::jit::Module& module) {
    std::vector<Tensor> result;
    for (auto p : module.parameters()) result.push_back(p);
    return result;
}
Tensor Forward(torch::jit::Module& module, const Tensor& input) {
    return module.forward({input}).toTensor();
}
void Finite(const Tensor& value, const char* what) {
    TORCH_CHECK(torch::isfinite(value).all().item<bool>(), "Nonfinite ", what);
}

struct Distribution {
    Tensor actions, logp, entropy;
};
Distribution Policy(torch::jit::Module& actor, const Tensor& obs, Tensor actions = {}) {
    auto logits = Forward(actor, obs);
    auto logp = logits.log_softmax(-1);
    if (!actions.defined()) actions = (logits - torch::empty_like(logits).exponential_().log()).argmax(-1);
    return {actions, logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1),
            -(logp.exp() * logp).sum(-1)};
}

// Matches TransitionRollout: real successors, causal stride, persistent chunk history,
// and no windows across an episode reset (including boundaries in skipped decisions).
struct History {
    int span, length, stride;
    int64_t tick = 0;
    Tensor ring, age;
    History(const Tensor& obs, int scene, int l, int s)
        : span((l - 1) * s), length(l), stride(s),
          ring(torch::full({span + 1, obs.size(0), scene}, NAN, obs.options())),
          age(torch::zeros({obs.size(0)}, obs.options().dtype(torch::kLong))) {
        ring[0].copy_(obs.slice(1, 0, scene));
    }
    Tensor Advance(const Tensor& final, const Tensor& next, const Tensor& done) {
        ++tick; age += 1;
        int slot = tick % (span + 1);
        ring[slot].copy_(final);
        std::vector<Tensor> frames;
        for (int j = length - 1; j >= 0; --j) {
            int distance = j * stride;
            int index = (tick - distance) % (span + 1);
            if (index < 0) index += span + 1;
            frames.push_back(ring[index].masked_fill((age < distance).unsqueeze(-1), NAN));
        }
        auto window = torch::stack(frames, 1);
        ring[slot].copy_(torch::where(done.unsqueeze(-1), next, final));
        age.masked_fill_(done, 0);
        return window;
    }
};

Tensor Expert(const Tensor& frames, const Tensor& ends, const Tensor& lengths,
              int count, int length, int stride) {
    auto choices = torch::randint(ends.size(0), {count}, ends.options());
    auto offsets = torch::arange(1 - length, 1, ends.options());
    auto selected = ends.index_select(0, choices).unsqueeze(1) + offsets * stride;
    auto result = frames.index_select(0, selected.clamp_min(0).flatten())
                      .view({count, length, frames.size(1)});
    auto padding = offsets.unsqueeze(0) < (1 - lengths.index_select(0, choices)).unsqueeze(1);
    return result.masked_fill(padding.unsqueeze(-1), NAN);
}

Tensor Reward(const Tensor& logits, const json& c) {
    if (c["discriminator_forward_kl_reward"].get<bool>()) {
        float cap = c["discriminator_ratio_cap"];
        return (-logits).clamp_max(std::log(cap)).exp().clamp_max(cap).mean(-1) - 1;
    }
    if (c["discriminator_log_reward"].get<bool>()) return -torch::softplus(logits).mean(-1);
    if (c["amp"].get<bool>()) return (1 - .25 * (logits + 1).square()).clamp_min(0).mean(-1);
    return torch::softplus(-logits).mean(-1);
}

void GAE(const Tensor& rewards, const Tensor& values, const Tensor& timeoutValues,
         const Tensor& done, Tensor next, double gamma, double lambda,
         Tensor& advantages, Tensor& returns) {
    auto advantage = torch::zeros_like(next);
    for (int t = rewards.size(0) - 1; t >= 0; --t) {
        auto live = (~done[t]).to(torch::kFloat32);
        auto delta = rewards[t] + gamma * timeoutValues[t] + gamma * live * next - values[t];
        advantage = delta + gamma * lambda * live * advantage;
        advantages[t].copy_(advantage);
        next = values[t];
    }
    returns.copy_(advantages + values);
}

struct PPOLoss { Tensor loss, policy, value, kl; };
PPOLoss Objective(const Distribution& pi, const Tensor& value, const Tensor& oldLogp,
                  Tensor advantage, const Tensor& returns, const Tensor& oldValues,
                  const Tensor& scale, const Tensor& entropy, const json& c) {
    auto logRatio = pi.logp - oldLogp, ratio = logRatio.exp();
    advantage = (advantage - advantage.mean()) / advantage.std(false).clamp_min(1e-8);
    float clip = c["clip_coef"];
    auto policy = torch::maximum(-advantage * ratio,
                                -advantage * ratio.clamp(1 - clip, 1 + clip)).mean();
    auto error = ((value - returns) / scale).square();
    if (c["clip_value_loss"].get<bool>()) {
        auto radius = scale * c["value_clip_coef"].get<float>();
        auto clipped = oldValues + (value - oldValues).clamp(-radius, radius);
        error = torch::maximum(error, ((clipped - returns) / scale).square());
    }
    auto valueLoss = .5 * error.mean();
    return {policy + c["vf_coef"].get<double>() * valueLoss -
                (pi.entropy.mean(0) * entropy).sum(), policy, valueLoss,
            ((ratio.detach() - 1) - logRatio.detach()).mean()};
}

// Native numerical checks run against Python-generated reference fixtures.
void Check(const fs::path& directory) {
    torch::set_num_threads(1);
    auto fixture = torch::jit::load((directory / "checks.jit").string());
    auto get = [&](const char* key) { return fixture.attr(key).toTensor(); };
    auto obs = get("observations");
    History history(obs[0], obs.size(-1), 3, 2);
    std::vector<Tensor> windows;
    auto done = get("done");
    for (int t = 0; t < done.size(0); ++t)
        windows.push_back(history.Advance(get("finals")[t], obs[t + 1], done[t]));
    auto actual = torch::stack(windows);
    TORCH_CHECK(torch::equal(torch::isnan(actual), torch::isnan(get("windows"))), "history padding");
    TORCH_CHECK(torch::allclose(torch::nan_to_num(actual), torch::nan_to_num(get("windows"))), "history values");
    Tensor advantages = torch::empty_like(get("rewards")), returns = torch::empty_like(advantages);
    GAE(get("rewards"), get("values"), get("timeouts"), done, get("next_value"),
        .97, .91, advantages, returns);
    TORCH_CHECK(torch::allclose(advantages, get("advantages"), 1e-5, 1e-6), "GAE parity");
    json c = {{"discriminator_forward_kl_reward", true}, {"discriminator_ratio_cap", 10.0}};
    TORCH_CHECK(torch::allclose(Reward(get("logits"), c), get("ratio_reward")), "reward parity");
    json config; std::ifstream(directory / "config.json") >> config;
    auto actor = torch::jit::load((directory / "actor.jit").string());
    auto critic = torch::jit::load((directory / "critic.jit").string());
    auto pi = Policy(actor, get("ppo_obs"), get("ppo_actions"));
    auto loss = Objective(pi, Forward(critic, get("ppo_obs")).squeeze(-1), get("ppo_logp"),
        get("ppo_advantages"), get("ppo_returns"), get("ppo_values"), get("ppo_scale"),
        get("ppo_entropy"), config);
    TORCH_CHECK(torch::allclose(loss.loss, get("ppo_loss"), 1e-5, 1e-6), "PPO loss parity");
    loss.loss.backward();
    std::vector<Tensor> gradients;
    for (auto p : actor.parameters()) gradients.push_back(p.grad().flatten());
    for (auto p : critic.parameters()) gradients.push_back(p.grad().flatten());
    TORCH_CHECK(torch::allclose(torch::cat(gradients), get("ppo_gradients"), 1e-4, 2e-6), "PPO gradient parity");
    auto reset = get("reset_frames"), internal = get("reset_internal");
    Environments env(config, reset.data_ptr<float>(), internal.data_ptr<float>(), 1);
    auto observation = torch::empty({config["n_sim"].get<int>() * 2 * config["team_size"].get<int>(),
                                     config["observation_size"].get<int>()});
    env.Observe(observation.data_ptr<float>());
    torch::jit::Module output("Probe");
    output.register_buffer("observations", observation.clone());
    auto final = torch::empty_like(observation), next = torch::empty_like(observation);
    auto flags = torch::empty({observation.size(0)}, torch::kUInt8), terminal = torch::empty_like(flags);
    auto scores = torch::empty({observation.size(0)});
    auto controls = get("controls");
    env.Step(controls.data_ptr<int64_t>(), next.data_ptr<float>(), final.data_ptr<float>(),
             flags.data_ptr<uint8_t>(), terminal.data_ptr<uint8_t>(), scores.data_ptr<float>());
    output.register_buffer("final", final);
    output.register_buffer("next", next);
    output.register_buffer("done", flags);
    output.save((directory / "probe.jit").string());
    // A ball already behind the positive goal line must produce a blue goal,
    // paired terminal flags, and opposite signed rewards for the two teams.
    auto goalReset = reset.clone();
    goalReset[0][0] = 0; goalReset[0][1] = .9; goalReset[0][2] = .1;
    goalReset.slice(1, 3, 9).zero_();
    Environments goalEnv(config, goalReset.data_ptr<float>(), internal.data_ptr<float>(), 1);
    goalEnv.Step(controls.data_ptr<int64_t>(), next.data_ptr<float>(), final.data_ptr<float>(),
                flags.data_ptr<uint8_t>(), terminal.data_ptr<uint8_t>(), scores.data_ptr<float>());
    TORCH_CHECK(terminal.all().item<bool>() && flags.all().item<bool>(), "goal terminal flags");
    int team = config["team_size"];
    auto signedScores = scores.view({-1, 2 * team});
    TORCH_CHECK((signedScores.slice(1, 0, team) == 1).all().item<bool>() &&
                (signedScores.slice(1, team) == -1).all().item<bool>(), "goal team signs");
    std::cout << "Native history/reset, timeout GAE, forward-KL, PPO loss/gradient checks passed.\n";
}

void Train(const fs::path& directory) {
    json c; std::ifstream(directory / "config.json") >> c;
    torch::set_num_threads(c["torch_threads"]); torch::set_num_interop_threads(1);
    torch::manual_seed(c["seed"]);
    torch::Device device(c["device"].get<std::string>());
    // Same TF32 setting as --matmul-precision in the Python trainer.
    at::globalContext().setFloat32MatmulPrecision(c["matmul_precision"].get<std::string>());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    int sims = c["n_sim"], cars = 2 * c["team_size"].get<int>(), agents = sims * cars;
    int steps = c["rollout"], scene = c["scene_size"], obsSize = c["observation_size"];
    int length = c["discriminator_window_length"], stride = c["discriminator_decision_stride"];
    int64_t batchSize = int64_t(steps) * agents;
    auto actor = torch::jit::load((directory / "actor.jit").string(), device);
    auto critic = torch::jit::load((directory / "critic.jit").string(), device);
    auto disc = torch::jit::load((directory / "discriminator.jit").string(), device);
    auto target = torch::jit::load((directory / "discriminator.jit").string(), device);
    auto ap = Parameters(actor), cp = Parameters(critic), dp = Parameters(disc), tp = Parameters(target);
    auto all = ap; all.insert(all.end(), cp.begin(), cp.end());
    torch::optim::Adam ppo(all, torch::optim::AdamOptions(c["learning_rate"].get<double>())
                                             .eps(c["adam_eps"].get<double>()));
    torch::optim::Adam optimizer(dp, torch::optim::AdamOptions(c["discriminator_learning_rate"].get<double>()));
    auto data = torch::jit::load((directory / "data.jit").string());
    Sync(device); auto dataStarted = Clock::now();
    auto frames = data.attr("frames").toTensor().to(device);
    auto ends = data.attr("ends").toTensor().to(device);
    auto lengths = data.attr("lengths").toTensor().to(device);
    Sync(device); double dataTransferSeconds = Seconds(dataStarted);
    int64_t expertBytes = frames.nbytes() + ends.nbytes() + lengths.nbytes();
    auto resets = data.attr("reset_frames").toTensor().contiguous();
    auto internals = data.attr("reset_internal").toTensor().contiguous();
    Environments env(c, resets.numel() ? resets.data_ptr<float>() : nullptr,
                     internals.numel() ? internals.data_ptr<float>() : nullptr,
                     c["reset_states"]);
    auto hostOptions = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device.is_cuda());
    auto hostObs = torch::empty({agents, obsSize}, hostOptions);
    auto hostFinal = torch::empty_like(hostObs), hostNext = torch::empty_like(hostObs);
    auto hostDone = torch::empty({agents}, hostOptions.dtype(torch::kUInt8));
    auto hostTerm = torch::empty_like(hostDone), hostScore = torch::empty({agents}, hostOptions);
    env.Observe(hostObs.data_ptr<float>());
    Tensor obs = hostObs.to(device);
    // On CPU to(device) aliases host memory: clone the first observation explicitly.
    obs = obs.clone();
    History history(obs, scene, length, stride);
    auto observations = torch::empty({steps, agents, obsSize}, options);
    auto actions = torch::empty({steps, agents, 7}, options.dtype(torch::kLong));
    auto values = torch::empty({steps, agents}, options), oldLogp = torch::empty_like(values);
    auto done = torch::empty({steps, agents}, options.dtype(torch::kBool));
    auto terminated = torch::empty_like(done);
    auto timeouts = torch::empty_like(values), scores = torch::empty_like(values);
    auto windows = torch::empty({steps, agents, length, scene}, options);
    auto advantages = torch::empty_like(values), returns = torch::empty_like(values);
    auto noiseScale = torch::zeros({scene}, torch::kFloat32);
    auto mirror = torch::ones({scene}, torch::kFloat32);
    auto ns = noiseScale.accessor<float, 1>(), ms = mirror.accessor<float, 1>();
    const float ballSigns[9] = {-1,1,1,-1,1,1,1,-1,-1};
    const float carSigns[21] = {-1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,1,1,1};
    for (int i = 0; i < 9; ++i) { ns[i] = c["discriminator_motion_noise_scale"]; ms[i] = ballSigns[i]; }
    for (int j = 0; j < cars; ++j) for (int i = 0; i < 21; ++i) {
        ms[9 + j * 21 + i] = carSigns[i];
        if (i < 9) ns[9 + j * 21 + i] = c["discriminator_motion_noise_scale"];
    }
    noiseScale = noiseScale.to(device); mirror = mirror.to(device);
    auto entropyCoefficients = torch::tensor(c["entropy_coefficients"].get<std::vector<float>>(), options);
    int capacity = c["discriminator_history_capacity"], poolSize = 0, cursor = 0;
    Tensor pool;
    if (capacity) pool = torch::empty({capacity, length, scene}, options);
    int64_t step = 0, iteration = 0, measuredSteps = 0;
    double measuredSeconds = 0, collectionTotal = 0, discTotal = 0, ppoTotal = 0, rewardTotal = 0;
    bool profileTransfers = device.is_cuda() && c.value("profile_transfers", false);
    double h2dTotal = 0, d2hTotal = 0, simulationTotal = 0, inferenceTotal = 0;
    // Two float observation matrices plus uint8 done/terminal and float score.
    int64_t h2dBytesPerStep = device.is_cuda() ? 2 * obsSize * sizeof(float) + 6 : 0;
    int64_t d2hBytesPerStep = device.is_cuda() ? 7 * sizeof(int64_t) : 0;
    std::ofstream metrics(directory / "metrics.jsonl");
    auto save = [&] {
        actor.save((directory / "latest-actor.jit").string());
        critic.save((directory / "latest-critic.jit").string());
        disc.save((directory / "latest-discriminator.jit").string());
        target.save((directory / "latest-target.jit").string());
        torch::save(ppo, (directory / "latest-ppo-optimizer.pt").string());
        torch::save(optimizer, (directory / "latest-discriminator-optimizer.pt").string());
        std::ofstream(directory / "latest.json") << json{{"step", step}, {"iteration", iteration},
            {"format", "goddard-native-gaifo-v1"}}.dump(2) << '\n';
    };
    while (step < c["timesteps"].get<int64_t>()) {
        Sync(device); auto started = Clock::now();
        double h2dSeconds = 0, d2hSeconds = 0, simulationSeconds = 0, inferenceSeconds = 0;
        double gamma = c["gamma"];
        if (!c["gamma_end"].is_null()) {
            double progress = std::min(double(step) / c["gamma_anneal_steps"].get<double>(), 1.0);
            gamma += progress * (c["gamma_end"].get<double>() - gamma);
        }
        double rewardGamma = std::pow(gamma, c["frameskip"].get<double>() / 4);
        auto fraction = std::min(double(step) / c["timesteps"].get<double>(), 1.0);
        auto lr = c["learning_rate"].get<double>() + fraction *
                  (c["learning_rate_end"].get<double>() - c["learning_rate"].get<double>());
        static_cast<torch::optim::AdamOptions&>(ppo.param_groups()[0].options()).lr(lr);
        {
            torch::NoGradGuard guard;
            for (int t = 0; t < steps; ++t) {
                if (profileTransfers) Sync(device);
                auto inferenceStarted = Clock::now();
                observations[t].copy_(obs);
                auto policy = Policy(actor, obs);
                actions[t].copy_(policy.actions); oldLogp[t].copy_(policy.logp);
                values[t].copy_(Forward(critic, obs).squeeze(-1));
                if (profileTransfers) {
                    Sync(device);
                    inferenceSeconds += Seconds(inferenceStarted);
                }
                auto copyStarted = Clock::now();
                auto hostActions = policy.actions.to(torch::kCPU).contiguous();
                if (profileTransfers) d2hSeconds += Seconds(copyStarted);
                auto simulationStarted = Clock::now();
                env.Step(hostActions.data_ptr<int64_t>(), hostNext.data_ptr<float>(),
                         hostFinal.data_ptr<float>(), hostDone.data_ptr<uint8_t>(),
                         hostTerm.data_ptr<uint8_t>(), hostScore.data_ptr<float>());
                if (profileTransfers) simulationSeconds += Seconds(simulationStarted);
                if (profileTransfers) Sync(device);
                copyStarted = Clock::now();
                auto final = hostFinal.to(device);
                auto next = hostNext.to(device);
                // CPU .to() aliases the reusable host buffers; CUDA already allocates.
                if (!device.is_cuda()) { final = final.clone(); next = next.clone(); }
                done[t].copy_(hostDone); terminated[t].copy_(hostTerm); scores[t].copy_(hostScore);
                if (profileTransfers) { Sync(device); h2dSeconds += Seconds(copyStarted); }
                windows[t].copy_(history.Advance(final.slice(1, 0, scene),
                                                next.slice(1, 0, scene), done[t]));
                timeouts[t].zero_();
                bool truncated = false;
                for (int i = 0; i < agents; ++i)
                    truncated |= hostDone.data_ptr<uint8_t>()[i] && !hostTerm.data_ptr<uint8_t>()[i];
                if (truncated) timeouts[t].copy_(Forward(critic, final).squeeze(-1) *
                                               (done[t] & ~terminated[t]));
                obs = next;
            }
        }
        Sync(device); double collection = Seconds(started); auto stage = Clock::now();
        auto flatWindows = windows.flatten(0, 1);
        auto valid = (~terminated.flatten()).nonzero().flatten();
        double discLoss = 0; int discUpdates = 0;
        int sourceBatch = c["discriminator_batch_size"], micro = c["discriminator_update_microbatch_size"];
        for (int epoch = 0; epoch < c["discriminator_updates"].get<int>(); ++epoch) {
            auto order = valid.index_select(0, torch::randperm(valid.numel(), valid.options()));
            for (auto indices : order.split(sourceBatch)) {
                int count = indices.numel();
                if (!count) continue;
                auto generated = flatWindows.index_select(0, indices);
                int historical = poolSize ? int(count * c["discriminator_history_mix_fraction"].get<double>()) : 0;
                if (historical) generated.slice(0, 0, historical).copy_(pool.index_select(0,
                    torch::randint(poolSize, {historical}, indices.options())));
                auto expert = Expert(frames, ends, lengths, count, length, c["expert_decision_stride"]);
                auto padding = torch::isnan(generated.select(-1, 0)) | torch::isnan(expert.select(-1, 0));
                auto input = torch::cat({generated, expert}).masked_fill(
                    torch::cat({padding, padding}).unsqueeze(-1), NAN);
                if (c["discriminator_motion_noise_scale"].get<float>() > 0)
                    input += torch::randn_like(input) * noiseScale;
                if (c["discriminator_mirror"].get<bool>())
                    input *= torch::where(torch::rand({2 * count, 1, 1}, options) < .5,
                                          mirror.view({1, 1, scene}), torch::ones_like(mirror));
                optimizer.zero_grad();
                Tensor total = torch::zeros({}, options);
                for (int start = 0; start < count; start += micro) {
                    int stop = std::min(start + micro, count), n = stop - start;
                    auto x = torch::cat({input.slice(0, start, stop), input.slice(0, count + start, count + stop)});
                    auto z = Forward(disc, x);
                    auto labels = torch::cat({torch::ones({n, 1}, options), torch::zeros({n, 1}, options)});
                    Tensor loss;
                    if (c["amp"].get<bool>()) loss = (z - (2 * labels - 1)).square().mean();
                    else loss = 2 * (torch::softplus(z) - labels * z).mean();
                    loss *= double(n) / count;
                    loss.backward(); total += loss.detach();
                }
                double norm = torch::nn::utils::clip_grad_norm_(dp, c["max_grad_norm"].get<double>(), 2, true);
                (void)norm; Finite(total, "discriminator loss"); optimizer.step();
                if (c["target_discriminator"].get<bool>()) {
                    torch::NoGradGuard guard;
                    for (size_t i = 0; i < tp.size(); ++i) tp[i].lerp_(dp[i], c["target_discriminator_tau"].get<double>());
                }
                discLoss += total.item<double>(); ++discUpdates;
            }
        }
        if (capacity && iteration % c["discriminator_history_interval"].get<int>() == 0) {
            torch::NoGradGuard guard;
            int count = std::min({capacity, c["discriminator_history_add_size"].get<int>(), int(valid.numel())});
            auto selected = valid.index_select(0, torch::randperm(valid.numel(), valid.options()).slice(0, 0, count));
            auto slots = (torch::arange(count, valid.options()) + cursor) % capacity;
            pool.index_copy_(0, slots, flatWindows.index_select(0, selected));
            cursor = (cursor + count) % capacity; poolSize = std::min(capacity, poolSize + count);
        }
        Sync(device); double discriminatorTime = Seconds(stage); stage = Clock::now();
        auto rewards = torch::empty_like(values);
        {
            torch::NoGradGuard guard;
            auto& model = c["target_discriminator"].get<bool>() ? target : disc;
            int infer = c["discriminator_inference_microbatch_size"];
            for (int64_t start = 0; start < batchSize; start += infer) {
                int64_t stop = std::min(start + infer, batchSize);
                rewards.flatten().slice(0, start, stop).copy_(Reward(Forward(model,
                    flatWindows.slice(0, start, stop)), c));
            }
            rewards.masked_fill_(terminated, 0);
            rewards += scores * c["goal_reward_weight"].get<double>();
            Finite(rewards, "rewards");
            GAE(rewards, values, timeouts, done, Forward(critic, obs).squeeze(-1),
                rewardGamma, c["reward_gae_lambda"], advantages, returns);
        }
        Sync(device); double rewardTime = Seconds(stage); stage = Clock::now();
        auto valueScale = c["normalize_value_loss"].get<bool>() ? returns.std(false).clamp_min(1) : torch::ones({}, options);
        double policyLossSum = 0, valueLossSum = 0; int updates = 0;
        for (int epoch = 0; epoch < c["update_epochs"].get<int>(); ++epoch) {
            double epochKL = 0;
            for (auto indices : torch::randperm(batchSize, options.dtype(torch::kLong)).split(c["minibatch_size"].get<int>())) {
                auto select = [&](const Tensor& x) { return x.flatten(0, 1).index_select(0, indices); };
                ppo.zero_grad();
                auto x = select(observations);
                auto pi = Policy(actor, x, select(actions));
                auto value = Forward(critic, x).squeeze(-1);
                auto losses = Objective(pi, value, select(oldLogp), select(advantages),
                    select(returns), select(values), valueScale, entropyCoefficients, c);
                losses.loss.backward();
                torch::nn::utils::clip_grad_norm_(ap, c["max_grad_norm"].get<double>(), 2, true);
                torch::nn::utils::clip_grad_norm_(cp, c["critic_max_grad_norm"].is_null() ?
                    c["max_grad_norm"].get<double>() : c["critic_max_grad_norm"].get<double>(), 2, true);
                Finite(losses.loss, "PPO loss"); ppo.step();
                epochKL += losses.kl.item<double>() * indices.numel();
                policyLossSum += losses.policy.item<double>(); valueLossSum += losses.value.item<double>(); ++updates;
            }
            if (!c["target_kl"].is_null() && epochKL / batchSize > c["target_kl"].get<double>()) break;
        }
        Sync(device); double ppoTime = Seconds(stage), elapsed = Seconds(started);
        int64_t previous = step; step += batchSize; ++iteration;
        json record = {{"iteration", iteration}, {"step", step}, {"seconds", elapsed},
            {"collection_seconds", collection}, {"discriminator_seconds", discriminatorTime},
            {"reward_gae_seconds", rewardTime}, {"ppo_seconds", ppoTime},
            {"agent_steps_per_second", batchSize / elapsed},
            {"agent_physics_ticks_per_second", batchSize * c["frameskip"].get<int>() / elapsed},
            {"arena_physics_ticks_per_second", batchSize * c["frameskip"].get<int>() / elapsed / cars},
            {"collection_agent_steps_per_second", batchSize / collection},
            {"h2d_bytes", batchSize * h2dBytesPerStep},
            {"d2h_bytes", batchSize * d2hBytesPerStep},
            {"transfer_profiling", profileTransfers},
            {"discriminator_loss", discLoss / std::max(discUpdates, 1)},
            {"policy_loss", policyLossSum / updates}, {"value_loss", valueLossSum / updates},
            {"discriminator_minibatches", discUpdates}, {"ppo_minibatches", updates},
            {"warmup", iteration <= c["warmup_iterations"].get<int>()}};
        if (profileTransfers) {
            record["h2d_seconds"] = h2dSeconds;
            record["d2h_seconds"] = d2hSeconds;
            record["simulation_observation_seconds"] = simulationSeconds;
            record["inference_storage_seconds"] = inferenceSeconds;
        }
        metrics << record.dump() << '\n'; metrics.flush();
        std::cout << record.dump() << std::endl;
        if (iteration > c["warmup_iterations"].get<int>()) {
            measuredSteps += batchSize; measuredSeconds += elapsed;
            collectionTotal += collection; discTotal += discriminatorTime;
            rewardTotal += rewardTime; ppoTotal += ppoTime;
            h2dTotal += h2dSeconds; d2hTotal += d2hSeconds;
            simulationTotal += simulationSeconds; inferenceTotal += inferenceSeconds;
        }
        bool due = !c["save_every"].is_null() ? iteration % c["save_every"].get<int>() == 0 :
            previous / c["save_every_steps"].get<int64_t>() != step / c["save_every_steps"].get<int64_t>();
        if (due) save();
    }
    save();
    json summary = {{"backend", "rocketsim-cpp-libtorch"}, {"device", device.str()},
        {"measured_agent_steps", measuredSteps}, {"measured_seconds", measuredSeconds},
        {"expert_bank_bytes", expertBytes},
        {"expert_startup_transfer_seconds", dataTransferSeconds},
        {"reset_bank_host_bytes", resets.nbytes() + internals.nbytes()},
        {"transfer_profiling", profileTransfers},
        {"h2d_bytes", measuredSteps * h2dBytesPerStep},
        {"d2h_bytes", measuredSteps * d2hBytesPerStep},
        {"warmup_iterations", c["warmup_iterations"]}, {"collection_seconds", collectionTotal},
        {"discriminator_seconds", discTotal}, {"reward_gae_seconds", rewardTotal}, {"ppo_seconds", ppoTotal},
        {"agent_steps_per_second", measuredSeconds ? measuredSteps / measuredSeconds : 0},
        {"agent_physics_ticks_per_second", measuredSeconds ? measuredSteps * c["frameskip"].get<int>() / measuredSeconds : 0},
        {"arena_physics_ticks_per_second", measuredSeconds ? measuredSteps * c["frameskip"].get<int>() / measuredSeconds / cars : 0},
        {"timing_excludes", "build, export, data/mesh initialization, checkpoint and metrics writes"}};
    if (profileTransfers) {
        summary["h2d_seconds"] = h2dTotal;
        summary["d2h_seconds"] = d2hTotal;
        summary["simulation_observation_seconds"] = simulationTotal;
        summary["inference_storage_seconds"] = inferenceTotal;
    }
    std::ofstream(directory / "summary.json") << summary.dump(2) << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::runtime_error("usage: gaifo_native RUN_DIR [--check]");
        if (argc == 3 && std::string(argv[2]) == "--check") Check(argv[1]);
        else Train(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
