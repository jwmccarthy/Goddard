const controls = document.querySelector('#controls-info')
const boostGauge = document.querySelector('#boost-gauge')
const boostValue = document.querySelector('#boost-value')

export function updateHud(car, state, inputs) {
  controls.hidden = !car || !inputs
  if (inputs && car) {
    controls.querySelectorAll('[data-axis]').forEach((axis) => {
      const value = Math.max(-1, Math.min(1, inputs[axis.dataset.axis]))
      axis.querySelector('i').style.left = `${50 + 50 * value}%`
    })
    controls.querySelectorAll('[data-button]').forEach((button) => {
      button.classList.toggle('active', inputs[button.dataset.button])
    })
  }

  boostGauge.hidden = !car || !Number.isFinite(state?.boost)
  if (boostGauge.hidden) return
  const boost = Math.round(Math.max(0, Math.min(100, state.boost)))
  boostGauge.style.setProperty('--boost', `${boost}%`)
  boostGauge.setAttribute('aria-valuenow', boost)
  boostValue.textContent = boost
}
