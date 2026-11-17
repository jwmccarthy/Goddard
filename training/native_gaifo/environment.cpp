#include "environment.h"
#include <RocketSim.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

using namespace RocketSim;
namespace {
const Vec POS_SCALE(4108, 6000, 2076);
const std::array<Vec, 34> PADS = {{
    {-3584,0,73},{3584,0,73},{-3072,4096,73},{3072,4096,73},
    {-3072,-4096,73},{3072,-4096,73},{0,-4240,70},{-1792,-4184,70},
    {1792,-4184,70},{-940,-3308,70},{940,-3308,70},{0,-2816,70},
    {-3584,-2484,70},{3584,-2484,70},{-1788,-2300,70},{1788,-2300,70},
    {-2048,-1036,70},{0,-1024,70},{2048,-1036,70},{-1024,0,70},{1024,0,70},
    {-2048,1036,70},{0,1024,70},{2048,1036,70},{-1788,2300,70},{1788,2300,70},
    {-3584,2484,70},{3584,2484,70},{0,2816,70},{-940,3308,70},{940,3308,70},
    {-1792,4184,70},{1792,4184,70},{0,4240,70}
}};
const int INVERTED[34] = {1,0,5,4,3,2,33,32,31,30,29,28,27,26,25,24,23,
                         22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6};
Vec ReadVec(const float* p) { return {p[0], p[1], p[2]}; }
float Axis(int64_t a) { return a == 1 ? -1.f : a == 2 ? 1.f : 0.f; }

struct Game {
    std::unique_ptr<Arena> arena;
    std::vector<Car*> cars;
    std::array<BoostPad*, 34> pads;
    std::mt19937 rng;
    int ticks = 0, noTouch = 0, scorer = 0;
    Game(int teamSize, int seed) : arena(Arena::Create(GameMode::SOCCAR)), rng(seed) {
        arena->SetGoalScoreCallback([](Arena*, Team team, void* user) {
            auto& game = *static_cast<Game*>(user);
            if (!game.scorer) game.scorer = team == Team::BLUE ? 1 : -1;
        }, this);
        for (int i = 0; i < 2 * teamSize; ++i)
            cars.push_back(arena->AddCar(i < teamSize ? Team::BLUE : Team::ORANGE));
        for (int i = 0; i < 34; ++i) {
            auto& source = arena->GetBoostPads();
            pads[i] = *std::min_element(source.begin(), source.end(), [i](auto a, auto b) {
                return a->config.pos.DistSq2D(PADS[i]) < b->config.pos.DistSq2D(PADS[i]);
            });
            if (pads[i]->config.pos.DistSq2D(PADS[i]) > 1)
                throw std::runtime_error("RocketSim boost-pad layout does not match CARL");
        }
    }

    void Reset(const float* scenes, const float* internal, int64_t count, float fraction) {
        // Seed each arena independently; worker scheduling cannot change reset draws.
        arena->ResetToRandomKickoff(int(rng() & 0x7fffffff));
        if (count && std::generate_canonical<float, 24>(rng) < fraction) {
            int64_t index = std::uniform_int_distribution<int64_t>(0, count - 1)(rng);
            const float* row = scenes + index * (9 + 21 * cars.size());
            BallState ball;
            ball.pos = ReadVec(row) * POS_SCALE;
            ball.vel = ReadVec(row + 3) * 6000;
            ball.angVel = ReadVec(row + 6) * 6;
            arena->ball->SetState(ball);
            for (size_t i = 0; i < cars.size(); ++i) {
                auto p = row + 9 + 21 * i;
                auto q = internal + (index * cars.size() + i) * 19;
                CarState s;
                s.pos = ReadVec(p) * POS_SCALE;
                s.vel = ReadVec(p + 3) * 2300;
                s.angVel = ReadVec(p + 6) * 5.5f;
                s.rotMat = RotMat::LookAt(ReadVec(p + 9), ReadVec(p + 12));
                s.boost = std::clamp(p[15] * 100, 0.f, 100.f);
                s.isDemoed = p[17] > .5f;
                s.demoRespawnTimer = s.isDemoed ? 3.f : 0.f;
                s.isOnGround = q[0] != 0;
                s.airTimeSinceJump = q[1]; s.handbrakeVal = q[2];
                s.hasJumped = q[3] != 0; s.isJumping = q[4] != 0;
                s.lastControls.jump = q[5] != 0; s.jumpTime = q[6];
                s.hasDoubleJumped = q[7] != 0; s.hasFlipped = q[8] != 0;
                s.isFlipping = q[9] != 0; s.flipTime = q[10];
                s.isAutoFlipping = q[11] != 0; s.autoFlipTimer = q[12];
                s.autoFlipTorqueScale = q[13]; s.flipRelTorque = ReadVec(q + 14);
                s.timeSpentBoosting = q[17] ? std::max(q[18], 1.f / 120) : 0;
                // Replay internal rows do not store four separate wheel contacts.
                for (bool& contact : s.wheelsWithContact) contact = s.isOnGround && !s.isDemoed;
                cars[i]->SetState(s);
                cars[i]->controls = {};
            }
        }
        ticks = noTouch = scorer = 0;
    }

    void Observe(float* output, int teamSize, int obsSize) {
        const auto ball = arena->ball->GetState();
        std::vector<CarState> states;
        for (auto car : cars) states.push_back(car->GetState());
        for (int ego = 0; ego < int(cars.size()); ++ego) {
            float* out = output + ego * obsSize;
            int at = 0;
            bool invert = ego >= teamSize;
            auto vector = [&](Vec v, Vec scale) {
                if (invert) { v.x = -v.x; v.y = -v.y; }
                v /= scale;
                out[at++] = v.x; out[at++] = v.y; out[at++] = v.z;
            };
            auto isotropic = [&](Vec v, float scale) { vector(v, {scale, scale, scale}); };
            std::vector<int> order{ego};
            for (int i = invert ? teamSize : 0; i < (invert ? 2 * teamSize : teamSize); ++i)
                if (i != ego) order.push_back(i);
            for (int i = invert ? 0 : teamSize; i < (invert ? teamSize : 2 * teamSize); ++i)
                order.push_back(i);
            vector(ball.pos, POS_SCALE); isotropic(ball.vel, 6000); isotropic(ball.angVel, 6);
            for (int i : order) {
                const auto& s = states[i];
                vector(s.pos, POS_SCALE); isotropic(s.vel, 2300); isotropic(s.angVel, 5.5f);
                isotropic(s.rotMat.forward, 1); isotropic(s.rotMat.up, 1);
                out[at++] = s.boost / 100;
                out[at++] = s.isOnGround; out[at++] = s.isDemoed;
                out[at++] = s.hasFlipped; out[at++] = s.hasDoubleJumped;
                out[at++] = s.timeSpentBoosting > 0 && s.boost > 0;
            }
            const auto& s = states[ego];
            for (int i = 0; i < 34; ++i)
                out[at++] = pads[invert ? INVERTED[i] : i]->GetState().isActive;
            for (int i = 0; i < 34; ++i)
                out[at++] = s.pos.Dist(PADS[invert ? INVERTED[i] : i]) / 14692.54f;
            vector(ball.pos - s.pos, POS_SCALE * 2); isotropic(ball.vel - s.vel, 8300);
            for (size_t i = 1; i < order.size(); ++i) {
                vector(states[order[i]].pos - s.pos, POS_SCALE * 2);
                isotropic(states[order[i]].vel - s.vel, 4600);
            }
            float ownY = invert ? 5120.f : -5120.f;
            vector(Vec(0, ownY, 321.3875f) - ball.pos, POS_SCALE * 2);
            vector(Vec(0, -ownY, 321.3875f) - ball.pos, POS_SCALE * 2);
            bool available = s.HasFlipOrJump();
            out[at++] = available;
            out[at++] = available ? std::max(0.f, 1.25f - s.airTimeSinceJump) : 0;
            const auto& c = cars[ego]->controls;
            out[at++] = c.throttle; out[at++] = c.steer; out[at++] = c.yaw;
            out[at++] = c.pitch; out[at++] = c.roll; out[at++] = c.jump;
            out[at++] = c.boost; out[at++] = c.handbrake;
            out[at++] = s.lastControls.jump; out[at++] = s.hasJumped;
            out[at++] = s.isJumping; out[at++] = s.jumpTime;
            out[at++] = s.timeSpentBoosting > 0 ? std::max(0.f, .1f - s.timeSpentBoosting) : 0;
            out[at++] = s.isFlipping; out[at++] = s.flipTime;
            for (int axis = 0; axis < 3; ++axis) out[at++] = s.flipRelTorque[axis];
            out[at++] = s.handbrakeVal;
            for (bool contact : s.wheelsWithContact) out[at++] = contact && !s.isDemoed;
            out[at++] = s.isAutoFlipping; out[at++] = s.autoFlipTimer;
            out[at++] = s.autoFlipTorqueScale; out[at++] = s.demoRespawnTimer;
            if (at != obsSize) throw std::runtime_error("observation layout mismatch");
        }
    }
};
}

struct Environments::Impl {
    int teamSize, frameskip, maxTicks, timeout, obsSize, threads;
    float fraction;
    const float *resets, *internals;
    int64_t resetCount;
    std::vector<std::unique_ptr<Game>> games;
    Impl(const nlohmann::json& c, const float* r, const float* in, int64_t count)
        // CARL starts every episode tied and ends on the first goal. A live episode
        // therefore reaches its fixed overtime timeout after regulation expires.
        : teamSize(c["team_size"]), frameskip(c["frameskip"]),
          maxTicks(c["max_ticks"].get<int>() + c.value("overtime_timeout_ticks", 36000)),
          timeout(int(std::ceil(c["no_touch_timeout"].get<double>() * 120))),
          obsSize(c["observation_size"]), threads(c["threads"]), fraction(c["replay_reset_fraction"]),
          resets(r), internals(in), resetCount(count) {
        RocketSim::Init(c["collision_meshes"].get<std::string>(), true);
        for (int i = 0; i < c["n_sim"].get<int>(); ++i) {
            games.push_back(std::make_unique<Game>(teamSize, c["seed"].get<int>() + i * 7919));
            games.back()->Reset(resets, internals, resetCount, fraction);
        }
    }
};

Environments::Environments(const nlohmann::json& c, const float* r, const float* in, int64_t n)
    : impl(std::make_unique<Impl>(c, r, in, n)) {}
Environments::~Environments() = default;
void Environments::Observe(float* output) {
    auto& p = *impl;
    #pragma omp parallel for schedule(static) num_threads(p.threads)
    for (int i = 0; i < int(p.games.size()); ++i)
        p.games[i]->Observe(output + i * 2 * p.teamSize * p.obsSize, p.teamSize, p.obsSize);
}
void Environments::Step(const int64_t* actions, float* next, float* final,
                        uint8_t* done, uint8_t* terminated, float* scores) {
    auto& p = *impl;
    #pragma omp parallel for schedule(static) num_threads(p.threads)
    for (int i = 0; i < int(p.games.size()); ++i) {
        auto& g = *p.games[i];
        int n = 2 * p.teamSize, base = i * n;
        for (int j = 0; j < n; ++j) {
            auto a = actions + (base + j) * 7;
            auto& c = g.cars[j]->controls;
            c.steer = c.yaw = Axis(a[0]); c.pitch = Axis(a[1]);
            c.throttle = Axis(a[2]); c.handbrake = a[3] == 1;
            c.boost = a[4] == 1; c.roll = Axis(a[5]); c.jump = a[6] == 1;
        }
        auto before = g.arena->tickCount;
        g.arena->Step(p.frameskip);
        g.ticks += p.frameskip; g.noTouch += p.frameskip;
        uint64_t lastTouch = 0;
        bool touched = false;
        for (auto car : g.cars) {
            auto hit = car->GetState().ballHitInfo;
            if (hit.isValid && hit.tickCountWhenHit >= before) {
                lastTouch = std::max(lastTouch, hit.tickCountWhenHit);
                touched = true;
            }
        }
        if (touched) g.noTouch = int(g.arena->tickCount - lastTouch);
        bool goal = g.scorer != 0;
        bool boundary = goal || g.ticks >= p.maxTicks || (p.timeout && g.noTouch >= p.timeout);
        float score = float(g.scorer);
        g.Observe(final + base * p.obsSize, p.teamSize, p.obsSize);
        if (boundary) g.Reset(p.resets, p.internals, p.resetCount, p.fraction);
        if (boundary) g.Observe(next + base * p.obsSize, p.teamSize, p.obsSize);
        else std::copy(final + base * p.obsSize, final + (base + n) * p.obsSize,
                       next + base * p.obsSize);
        for (int j = 0; j < n; ++j) {
            done[base + j] = boundary; terminated[base + j] = goal;
            scores[base + j] = (j < p.teamSize ? score : -score);
        }
    }
}
