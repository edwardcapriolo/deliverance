package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the {@code FloatConversions} utility class, which provides methods
 * for performing conversions and operations involving floating-point values in
 * different formats, such as BFloat16 and IEEE Float16.
 * <p>
 * These tests verify the accuracy of conversions, rounding behaviors, handling of
 * special values like zeros, infinities, and NaN, as well as arithmetic operations for
 * IEEE Float16 values.
 */
class FloatConversionsTest {

    @Test
    void convertsBFloat16RawBitsToFloat32() {
        assertEquals(1.0f, FloatConversions.bFloat16ToFloat32((short) 0x3f80));
        assertEquals(-2.0f, FloatConversions.bFloat16ToFloat32((short) 0xc000));
        assertEquals(Float.POSITIVE_INFINITY,
                FloatConversions.bFloat16ToFloat32((short) 0x7f80));
        assertEquals(Float.NEGATIVE_INFINITY,
                FloatConversions.bFloat16ToFloat32((short) 0xff80));
    }

    @Test
    void roundsFloat32ToNearestEvenBFloat16() {
        assertEquals((short) 0x3f80, FloatConversions.float32ToBFloat16(1.0f));
        assertEquals((short) 0x3f80,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f808000)),
                "a midpoint rounds to the even lower mantissa");
        assertEquals((short) 0x3f82,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f818000)),
                "a midpoint rounds to the even upper mantissa");
        assertEquals((short) 0x3f81,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f808001)));
    }

    @Test
    void preservesBFloat16SignedZeroAndHandlesSpecialValues() {
        assertEquals((short) 0x0000, FloatConversions.float32ToBFloat16(0.0f));
        assertEquals((short) 0x8000, FloatConversions.float32ToBFloat16(-0.0f));
        assertEquals((short) 0x7f80, FloatConversions.float32ToBFloat16(Float.POSITIVE_INFINITY));
        assertEquals((short) 0xff80, FloatConversions.float32ToBFloat16(Float.NEGATIVE_INFINITY));
        assertEquals(FloatConversions.bFloat16NaN,
                FloatConversions.float32ToBFloat16(Float.NaN));
    }

    @Test
    void convertsAlternativeFloat16Values() {
        assertEquals(0.0f, FloatConversions.float16ToFloat32Alt((short) 0x0000));
        assertEquals(-0.0f, FloatConversions.float16ToFloat32Alt((short) 0x8000));
        assertEquals(1.0f, FloatConversions.float16ToFloat32Alt((short) 0x3c00));
        assertEquals(-2.0f, FloatConversions.float16ToFloat32Alt((short) 0xc000));
    }

    @Test
    void handlesIeeeFloat16SpecialValues() {
        assertEquals((short) 0x7c00,
                FloatConversions.addIeeeFloat16((short) 0x7c00, (short) 0x3c00));
        assertEquals((short) 0xfc00,
                FloatConversions.mulIeeeFloat16((short) 0xfc00, (short) 0x3c00));

        short nan = FloatConversions.addIeeeFloat16((short) 0x7c01, (short) 0x3c00);
        assertEquals((short) 0x7fff, nan);
        assertTrue(Float.isNaN(FloatConversions.bFloat16ToFloat32(FloatConversions.bFloat16NaN)));
        assertFalse(Float.isNaN(FloatConversions.bFloat16ToFloat32((short) 0x7f80)));
    }
}
