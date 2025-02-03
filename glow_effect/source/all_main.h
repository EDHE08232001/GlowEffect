#ifndef ALL_MAIN_H
#define ALL_MAIN_H

/**
 * @brief Callback for updating the key level parameter.
 *
 * @param newValue The new key level value.
 */
void bar_key_level_cb(int newValue);

/**
 * @brief Callback for updating the key scale parameter.
 *
 * @param newValue The new key scale value.
 */
void bar_key_scale_cb(int newValue);

/**
 * @brief Callback for updating the default scale parameter.
 *
 * @param newValue The new default scale value.
 */
void bar_default_scale_cb(int newValue);

/**
 * @brief Applies the glow effect on the current image using the loaded mask.
 */
void updateImage();

#endif // ALL_MAIN_H

